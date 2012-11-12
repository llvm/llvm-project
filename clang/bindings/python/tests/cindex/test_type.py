import gc

from clang.cindex import CursorKind
from clang.cindex import TranslationUnit
from clang.cindex import TypeKind
from nose.tools import raises
from .util import get_cursor
from .util import get_tu

kInput = """\

typedef int I;

struct teststruct {
  int a;
  I b;
  long c;
  unsigned long d;
  signed long e;
  const int f;
  int *g;
  int ***h;
};

"""

def test_a_struct():
    tu = get_tu(kInput)

    teststruct = get_cursor(tu, 'teststruct')
    assert teststruct is not None, "Could not find teststruct."
    fields = list(teststruct.get_children())
    assert all(x.kind == CursorKind.FIELD_DECL for x in fields)
    assert all(x.translation_unit is not None for x in fields)

    assert fields[0].spelling == 'a'
    assert not fields[0].type.is_const_qualified()
    assert fields[0].type.kind == TypeKind.INT
    assert fields[0].type.get_canonical().kind == TypeKind.INT

    assert fields[1].spelling == 'b'
    assert not fields[1].type.is_const_qualified()
    assert fields[1].type.kind == TypeKind.TYPEDEF
    assert fields[1].type.get_canonical().kind == TypeKind.INT
    assert fields[1].type.get_declaration().spelling == 'I'

    assert fields[2].spelling == 'c'
    assert not fields[2].type.is_const_qualified()
    assert fields[2].type.kind == TypeKind.LONG
    assert fields[2].type.get_canonical().kind == TypeKind.LONG

    assert fields[3].spelling == 'd'
    assert not fields[3].type.is_const_qualified()
    assert fields[3].type.kind == TypeKind.ULONG
    assert fields[3].type.get_canonical().kind == TypeKind.ULONG

    assert fields[4].spelling == 'e'
    assert not fields[4].type.is_const_qualified()
    assert fields[4].type.kind == TypeKind.LONG
    assert fields[4].type.get_canonical().kind == TypeKind.LONG

    assert fields[5].spelling == 'f'
    assert fields[5].type.is_const_qualified()
    assert fields[5].type.kind == TypeKind.INT
    assert fields[5].type.get_canonical().kind == TypeKind.INT

    assert fields[6].spelling == 'g'
    assert not fields[6].type.is_const_qualified()
    assert fields[6].type.kind == TypeKind.POINTER
    assert fields[6].type.get_pointee().kind == TypeKind.INT

    assert fields[7].spelling == 'h'
    assert not fields[7].type.is_const_qualified()
    assert fields[7].type.kind == TypeKind.POINTER
    assert fields[7].type.get_pointee().kind == TypeKind.POINTER
    assert fields[7].type.get_pointee().get_pointee().kind == TypeKind.POINTER
    assert fields[7].type.get_pointee().get_pointee().get_pointee().kind == TypeKind.INT

def test_references():
    """Ensure that a Type maintains a reference to a TranslationUnit."""

    tu = get_tu('int x;')
    children = list(tu.cursor.get_children())
    assert len(children) > 0

    cursor = children[0]
    t = cursor.type

    assert isinstance(t.translation_unit, TranslationUnit)

    # Delete main TranslationUnit reference and force a GC.
    del tu
    gc.collect()
    assert isinstance(t.translation_unit, TranslationUnit)

    # If the TU was destroyed, this should cause a segfault.
    decl = t.get_declaration()

constarrayInput="""
struct teststruct {
  void *A[2];
};
"""
def testConstantArray():
    tu = get_tu(constarrayInput)

    teststruct = get_cursor(tu, 'teststruct')
    assert teststruct is not None, "Didn't find teststruct??"
    fields = list(teststruct.get_children())
    assert fields[0].spelling == 'A'
    assert fields[0].type.kind == TypeKind.CONSTANTARRAY
    assert fields[0].type.get_array_element_type() is not None
    assert fields[0].type.get_array_element_type().kind == TypeKind.POINTER
    assert fields[0].type.get_array_size() == 2

def test_equal():
    """Ensure equivalence operators work on Type."""
    source = 'int a; int b; void *v;'
    tu = get_tu(source)

    a = get_cursor(tu, 'a')
    b = get_cursor(tu, 'b')
    v = get_cursor(tu, 'v')

    assert a is not None
    assert b is not None
    assert v is not None

    assert a.type == b.type
    assert a.type != v.type

    assert a.type != None
    assert a.type != 'foo'

def test_typekind_spelling():
    """Ensure TypeKind.spelling works."""
    tu = get_tu('int a;')
    a = get_cursor(tu, 'a')

    assert a is not None
    assert a.type.kind.spelling == 'Int'

def test_function_argument_types():
    """Ensure that Type.argument_types() works as expected."""
    tu = get_tu('void f(int, int);')
    f = get_cursor(tu, 'f')
    assert f is not None

    args = f.type.argument_types()
    assert args is not None
    assert len(args) == 2

    t0 = args[0]
    assert t0 is not None
    assert t0.kind == TypeKind.INT

    t1 = args[1]
    assert t1 is not None
    assert t1.kind == TypeKind.INT

    args2 = list(args)
    assert len(args2) == 2
    assert t0 == args2[0]
    assert t1 == args2[1]

@raises(TypeError)
def test_argument_types_string_key():
    """Ensure that non-int keys raise a TypeError."""
    tu = get_tu('void f(int, int);')
    f = get_cursor(tu, 'f')
    assert f is not None

    args = f.type.argument_types()
    assert len(args) == 2

    args['foo']

@raises(IndexError)
def test_argument_types_negative_index():
    """Ensure that negative indexes on argument_types Raises an IndexError."""
    tu = get_tu('void f(int, int);')
    f = get_cursor(tu, 'f')
    args = f.type.argument_types()

    args[-1]

@raises(IndexError)
def test_argument_types_overflow_index():
    """Ensure that indexes beyond the length of Type.argument_types() raise."""
    tu = get_tu('void f(int, int);')
    f = get_cursor(tu, 'f')
    args = f.type.argument_types()

    args[2]

@raises(Exception)
def test_argument_types_invalid_type():
    """Ensure that obtaining argument_types on a Type without them raises."""
    tu = get_tu('int i;')
    i = get_cursor(tu, 'i')
    assert i is not None

    i.type.argument_types()

def test_is_pod():
    """Ensure Type.is_pod() works."""
    tu = get_tu('int i; void f();')
    i = get_cursor(tu, 'i')
    f = get_cursor(tu, 'f')

    assert i is not None
    assert f is not None

    assert i.type.is_pod()
    assert not f.type.is_pod()

def test_function_variadic():
    """Ensure Type.is_function_variadic works."""

    source ="""
#include <stdarg.h>

void foo(int a, ...);
void bar(int a, int b);
"""

    tu = get_tu(source)
    foo = get_cursor(tu, 'foo')
    bar = get_cursor(tu, 'bar')

    assert foo is not None
    assert bar is not None

    assert isinstance(foo.type.is_function_variadic(), bool)
    assert foo.type.is_function_variadic()
    assert not bar.type.is_function_variadic()

def test_element_type():
    """Ensure Type.element_type works."""
    tu = get_tu('int i[5];')
    i = get_cursor(tu, 'i')
    assert i is not None

    assert i.type.kind == TypeKind.CONSTANTARRAY
    assert i.type.element_type.kind == TypeKind.INT

@raises(Exception)
def test_invalid_element_type():
    """Ensure Type.element_type raises if type doesn't have elements."""
    tu = get_tu('int i;')
    i = get_cursor(tu, 'i')
    assert i is not None
    i.element_type

def test_element_count():
    """Ensure Type.element_count works."""
    tu = get_tu('int i[5]; int j;')
    i = get_cursor(tu, 'i')
    j = get_cursor(tu, 'j')

    assert i is not None
    assert j is not None

    assert i.type.element_count == 5

    try:
        j.type.element_count
        assert False
    except:
        assert True

def test_is_volatile_qualified():
    """Ensure Type.is_volatile_qualified works."""

    tu = get_tu('volatile int i = 4; int j = 2;')

    i = get_cursor(tu, 'i')
    j = get_cursor(tu, 'j')

    assert i is not None
    assert j is not None

    assert isinstance(i.type.is_volatile_qualified(), bool)
    assert i.type.is_volatile_qualified()
    assert not j.type.is_volatile_qualified()

def test_is_restrict_qualified():
    """Ensure Type.is_restrict_qualified works."""

    tu = get_tu('struct s { void * restrict i; void * j; };')

    i = get_cursor(tu, 'i')
    j = get_cursor(tu, 'j')

    assert i is not None
    assert j is not None

    assert isinstance(i.type.is_restrict_qualified(), bool)
    assert i.type.is_restrict_qualified()
    assert not j.type.is_restrict_qualified()
