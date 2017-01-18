import gc
import os
import tempfile

from clang.cindex import CursorKind
from clang.cindex import Cursor
from clang.cindex import File
from clang.cindex import Index
from clang.cindex import SourceLocation
from clang.cindex import SourceRange
from clang.cindex import TranslationUnitSaveError
from clang.cindex import TranslationUnitLoadError
from clang.cindex import TranslationUnit
from .util import get_cursor
from .util import get_tu

kInputsDir = os.path.join(os.path.dirname(__file__), 'INPUTS')

def test_spelling():
    path = os.path.join(kInputsDir, 'hello.cpp')
    tu = TranslationUnit.from_source(path)
    assert tu.spelling == path

def test_cursor():
    path = os.path.join(kInputsDir, 'hello.cpp')
    tu = get_tu(path)
    c = tu.cursor
    assert isinstance(c, Cursor)
    assert c.kind is CursorKind.TRANSLATION_UNIT

def test_parse_arguments():
    path = os.path.join(kInputsDir, 'parse_arguments.c')
    tu = TranslationUnit.from_source(path, ['-DDECL_ONE=hello', '-DDECL_TWO=hi'])
    spellings = [c.spelling for c in tu.cursor.get_children()]
    assert spellings[-2] == 'hello'
    assert spellings[-1] == 'hi'

def test_reparse_arguments():
    path = os.path.join(kInputsDir, 'parse_arguments.c')
    tu = TranslationUnit.from_source(path, ['-DDECL_ONE=hello', '-DDECL_TWO=hi'])
    tu.reparse()
    spellings = [c.spelling for c in tu.cursor.get_children()]
    assert spellings[-2] == 'hello'
    assert spellings[-1] == 'hi'

def test_unsaved_files():
    tu = TranslationUnit.from_source('fake.c', ['-I./'], unsaved_files = [
            ('fake.c', """
#include "fake.h"
int x;
int SOME_DEFINE;
"""),
            ('./fake.h', """
#define SOME_DEFINE y
""")
            ])
    spellings = [c.spelling for c in tu.cursor.get_children()]
    assert spellings[-2] == 'x'
    assert spellings[-1] == 'y'

def test_unsaved_files_2():
    import StringIO
    tu = TranslationUnit.from_source('fake.c', unsaved_files = [
            ('fake.c', StringIO.StringIO('int x;'))])
    spellings = [c.spelling for c in tu.cursor.get_children()]
    assert spellings[-1] == 'x'

def normpaths_equal(path1, path2):
    """ Compares two paths for equality after normalizing them with
        os.path.normpath
    """
    return os.path.normpath(path1) == os.path.normpath(path2)

def test_includes():
    def eq(expected, actual):
        if not actual.is_input_file:
            return  normpaths_equal(expected[0], actual.source.name) and \
                    normpaths_equal(expected[1], actual.include.name)
        else:
            return normpaths_equal(expected[1], actual.include.name)

    src = os.path.join(kInputsDir, 'include.cpp')
    h1 = os.path.join(kInputsDir, "header1.h")
    h2 = os.path.join(kInputsDir, "header2.h")
    h3 = os.path.join(kInputsDir, "header3.h")
    inc = [(src, h1), (h1, h3), (src, h2), (h2, h3)]

    tu = TranslationUnit.from_source(src)
    for i in zip(inc, tu.get_includes()):
        assert eq(i[0], i[1])

def save_tu(tu):
    """Convenience API to save a TranslationUnit to a file.

    Returns the filename it was saved to.
    """
    _, path = tempfile.mkstemp()
    tu.save(path)

    return path

def test_save():
    """Ensure TranslationUnit.save() works."""

    tu = get_tu('int foo();')

    path = save_tu(tu)
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0
    os.unlink(path)

def test_save_translation_errors():
    """Ensure that saving to an invalid directory raises."""

    tu = get_tu('int foo();')

    path = '/does/not/exist/llvm-test.ast'
    assert not os.path.exists(os.path.dirname(path))

    try:
        tu.save(path)
        assert False
    except TranslationUnitSaveError as ex:
        expected = TranslationUnitSaveError.ERROR_UNKNOWN
        assert ex.save_error == expected

def test_load():
    """Ensure TranslationUnits can be constructed from saved files."""

    tu = get_tu('int foo();')
    assert len(tu.diagnostics) == 0
    path = save_tu(tu)

    assert os.path.exists(path)
    assert os.path.getsize(path) > 0

    tu2 = TranslationUnit.from_ast_file(filename=path)
    assert len(tu2.diagnostics) == 0

    foo = get_cursor(tu2, 'foo')
    assert foo is not None

    # Just in case there is an open file descriptor somewhere.
    del tu2

    os.unlink(path)

def test_index_parse():
    path = os.path.join(kInputsDir, 'hello.cpp')
    index = Index.create()
    tu = index.parse(path)
    assert isinstance(tu, TranslationUnit)

def test_get_file():
    """Ensure tu.get_file() works appropriately."""

    tu = get_tu('int foo();')

    f = tu.get_file('t.c')
    assert isinstance(f, File)
    assert f.name == 't.c'

    try:
        f = tu.get_file('foobar.cpp')
    except:
        pass
    else:
        assert False

def test_get_source_location():
    """Ensure tu.get_source_location() works."""

    tu = get_tu('int foo();')

    location = tu.get_location('t.c', 2)
    assert isinstance(location, SourceLocation)
    assert location.offset == 2
    assert location.file.name == 't.c'

    location = tu.get_location('t.c', (1, 3))
    assert isinstance(location, SourceLocation)
    assert location.line == 1
    assert location.column == 3
    assert location.file.name == 't.c'

def test_get_source_range():
    """Ensure tu.get_source_range() works."""

    tu = get_tu('int foo();')

    r = tu.get_extent('t.c', (1,4))
    assert isinstance(r, SourceRange)
    assert r.start.offset == 1
    assert r.end.offset == 4
    assert r.start.file.name == 't.c'
    assert r.end.file.name == 't.c'

    r = tu.get_extent('t.c', ((1,2), (1,3)))
    assert isinstance(r, SourceRange)
    assert r.start.line == 1
    assert r.start.column == 2
    assert r.end.line == 1
    assert r.end.column == 3
    assert r.start.file.name == 't.c'
    assert r.end.file.name == 't.c'

    start = tu.get_location('t.c', 0)
    end = tu.get_location('t.c', 5)

    r = tu.get_extent('t.c', (start, end))
    assert isinstance(r, SourceRange)
    assert r.start.offset == 0
    assert r.end.offset == 5
    assert r.start.file.name == 't.c'
    assert r.end.file.name == 't.c'

def test_get_tokens_gc():
    """Ensures get_tokens() works properly with garbage collection."""

    tu = get_tu('int foo();')
    r = tu.get_extent('t.c', (0, 10))
    tokens = list(tu.get_tokens(extent=r))

    assert tokens[0].spelling == 'int'
    gc.collect()
    assert tokens[0].spelling == 'int'

    del tokens[1]
    gc.collect()
    assert tokens[0].spelling == 'int'

    # May trigger segfault if we don't do our job properly.
    del tokens
    gc.collect()
    gc.collect() # Just in case.

def test_fail_from_source():
    path = os.path.join(kInputsDir, 'non-existent.cpp')
    try:
        tu = TranslationUnit.from_source(path)
    except TranslationUnitLoadError:
        tu = None
    assert tu == None

def test_fail_from_ast_file():
    path = os.path.join(kInputsDir, 'non-existent.ast')
    try:
        tu = TranslationUnit.from_ast_file(path)
    except TranslationUnitLoadError:
        tu = None
    assert tu == None
