import ctypes
import gc

from clang.cindex import CursorKind
from clang.cindex import TemplateArgumentKind
from clang.cindex import TranslationUnit
from clang.cindex import TypeKind
from .util import get_cursor
from .util import get_cursors
from .util import get_tu

kInput = """\
struct s0 {
  int a;
  int b;
};

struct s1;

void f0(int a0, int a1) {
  int l0, l1;

  if (a0)
    return;

  for (;;) {
    break;
  }
}
"""

def test_get_children():
    tu = get_tu(kInput)

    it = tu.cursor.get_children()
    tu_nodes = list(it)

    assert len(tu_nodes) == 3
    for cursor in tu_nodes:
        assert cursor.translation_unit is not None

    assert tu_nodes[0] != tu_nodes[1]
    assert tu_nodes[0].kind == CursorKind.STRUCT_DECL
    assert tu_nodes[0].spelling == 's0'
    assert tu_nodes[0].is_definition() == True
    assert tu_nodes[0].location.file.name == 't.c'
    assert tu_nodes[0].location.line == 1
    assert tu_nodes[0].location.column == 8
    assert tu_nodes[0].hash > 0
    assert tu_nodes[0].translation_unit is not None

    s0_nodes = list(tu_nodes[0].get_children())
    assert len(s0_nodes) == 2
    assert s0_nodes[0].kind == CursorKind.FIELD_DECL
    assert s0_nodes[0].spelling == 'a'
    assert s0_nodes[0].type.kind == TypeKind.INT
    assert s0_nodes[1].kind == CursorKind.FIELD_DECL
    assert s0_nodes[1].spelling == 'b'
    assert s0_nodes[1].type.kind == TypeKind.INT

    assert tu_nodes[1].kind == CursorKind.STRUCT_DECL
    assert tu_nodes[1].spelling == 's1'
    assert tu_nodes[1].displayname == 's1'
    assert tu_nodes[1].is_definition() == False

    assert tu_nodes[2].kind == CursorKind.FUNCTION_DECL
    assert tu_nodes[2].spelling == 'f0'
    assert tu_nodes[2].displayname == 'f0(int, int)'
    assert tu_nodes[2].is_definition() == True

def test_references():
    """Ensure that references to TranslationUnit are kept."""
    tu = get_tu('int x;')
    cursors = list(tu.cursor.get_children())
    assert len(cursors) > 0

    cursor = cursors[0]
    assert isinstance(cursor.translation_unit, TranslationUnit)

    # Delete reference to TU and perform a full GC.
    del tu
    gc.collect()
    assert isinstance(cursor.translation_unit, TranslationUnit)

    # If the TU was destroyed, this should cause a segfault.
    parent = cursor.semantic_parent

def test_canonical():
    source = 'struct X; struct X; struct X { int member; };'
    tu = get_tu(source)

    cursors = []
    for cursor in tu.cursor.get_children():
        if cursor.spelling == 'X':
            cursors.append(cursor)

    assert len(cursors) == 3
    assert cursors[1].canonical == cursors[2].canonical

def test_is_const_method():
    """Ensure Cursor.is_const_method works."""
    source = 'class X { void foo() const; void bar(); };'
    tu = get_tu(source, lang='cpp')

    cls = get_cursor(tu, 'X')
    foo = get_cursor(tu, 'foo')
    bar = get_cursor(tu, 'bar')
    assert cls is not None
    assert foo is not None
    assert bar is not None

    assert foo.is_const_method()
    assert not bar.is_const_method()

def test_is_converting_constructor():
    """Ensure Cursor.is_converting_constructor works."""
    source = 'class X { explicit X(int); X(double); X(); };'
    tu = get_tu(source, lang='cpp')

    xs = get_cursors(tu, 'X')

    assert len(xs) == 4
    assert xs[0].kind == CursorKind.CLASS_DECL
    cs = xs[1:]
    assert cs[0].kind == CursorKind.CONSTRUCTOR
    assert cs[1].kind == CursorKind.CONSTRUCTOR
    assert cs[2].kind == CursorKind.CONSTRUCTOR

    assert not cs[0].is_converting_constructor()
    assert cs[1].is_converting_constructor()
    assert not cs[2].is_converting_constructor()


def test_is_copy_constructor():
    """Ensure Cursor.is_copy_constructor works."""
    source = 'class X { X(); X(const X&); X(X&&); };'
    tu = get_tu(source, lang='cpp')

    xs = get_cursors(tu, 'X')
    assert xs[0].kind == CursorKind.CLASS_DECL
    cs = xs[1:]
    assert cs[0].kind == CursorKind.CONSTRUCTOR
    assert cs[1].kind == CursorKind.CONSTRUCTOR
    assert cs[2].kind == CursorKind.CONSTRUCTOR

    assert not cs[0].is_copy_constructor()
    assert cs[1].is_copy_constructor()
    assert not cs[2].is_copy_constructor()

def test_is_default_constructor():
    """Ensure Cursor.is_default_constructor works."""
    source = 'class X { X(); X(int); };'
    tu = get_tu(source, lang='cpp')

    xs = get_cursors(tu, 'X')
    assert xs[0].kind == CursorKind.CLASS_DECL
    cs = xs[1:]
    assert cs[0].kind == CursorKind.CONSTRUCTOR
    assert cs[1].kind == CursorKind.CONSTRUCTOR

    assert cs[0].is_default_constructor()
    assert not cs[1].is_default_constructor()

def test_is_move_constructor():
    """Ensure Cursor.is_move_constructor works."""
    source = 'class X { X(); X(const X&); X(X&&); };'
    tu = get_tu(source, lang='cpp')

    xs = get_cursors(tu, 'X')
    assert xs[0].kind == CursorKind.CLASS_DECL
    cs = xs[1:]
    assert cs[0].kind == CursorKind.CONSTRUCTOR
    assert cs[1].kind == CursorKind.CONSTRUCTOR
    assert cs[2].kind == CursorKind.CONSTRUCTOR

    assert not cs[0].is_move_constructor()
    assert not cs[1].is_move_constructor()
    assert cs[2].is_move_constructor()

def test_is_default_method():
    """Ensure Cursor.is_default_method works."""
    source = 'class X { X() = default; }; class Y { Y(); };'
    tu = get_tu(source, lang='cpp')

    xs = get_cursors(tu, 'X')
    ys = get_cursors(tu, 'Y')

    assert len(xs) == 2
    assert len(ys) == 2

    xc = xs[1]
    yc = ys[1]

    assert xc.is_default_method()
    assert not yc.is_default_method()

def test_is_mutable_field():
    """Ensure Cursor.is_mutable_field works."""
    source = 'class X { int x_; mutable int y_; };'
    tu = get_tu(source, lang='cpp')

    cls = get_cursor(tu, 'X')
    x_ = get_cursor(tu, 'x_')
    y_ = get_cursor(tu, 'y_')
    assert cls is not None
    assert x_ is not None
    assert y_ is not None

    assert not x_.is_mutable_field()
    assert y_.is_mutable_field()

def test_is_static_method():
    """Ensure Cursor.is_static_method works."""

    source = 'class X { static void foo(); void bar(); };'
    tu = get_tu(source, lang='cpp')

    cls = get_cursor(tu, 'X')
    foo = get_cursor(tu, 'foo')
    bar = get_cursor(tu, 'bar')
    assert cls is not None
    assert foo is not None
    assert bar is not None

    assert foo.is_static_method()
    assert not bar.is_static_method()

def test_is_pure_virtual_method():
    """Ensure Cursor.is_pure_virtual_method works."""
    source = 'class X { virtual void foo() = 0; virtual void bar(); };'
    tu = get_tu(source, lang='cpp')

    cls = get_cursor(tu, 'X')
    foo = get_cursor(tu, 'foo')
    bar = get_cursor(tu, 'bar')
    assert cls is not None
    assert foo is not None
    assert bar is not None

    assert foo.is_pure_virtual_method()
    assert not bar.is_pure_virtual_method()

def test_is_virtual_method():
    """Ensure Cursor.is_virtual_method works."""
    source = 'class X { virtual void foo(); void bar(); };'
    tu = get_tu(source, lang='cpp')

    cls = get_cursor(tu, 'X')
    foo = get_cursor(tu, 'foo')
    bar = get_cursor(tu, 'bar')
    assert cls is not None
    assert foo is not None
    assert bar is not None

    assert foo.is_virtual_method()
    assert not bar.is_virtual_method()

def test_is_scoped_enum():
    """Ensure Cursor.is_scoped_enum works."""
    source = 'class X {}; enum RegularEnum {}; enum class ScopedEnum {};'
    tu = get_tu(source, lang='cpp')

    cls = get_cursor(tu, 'X')
    regular_enum = get_cursor(tu, 'RegularEnum')
    scoped_enum = get_cursor(tu, 'ScopedEnum')
    assert cls is not None
    assert regular_enum is not None
    assert scoped_enum is not None

    assert not cls.is_scoped_enum()
    assert not regular_enum.is_scoped_enum()
    assert scoped_enum.is_scoped_enum()

def test_underlying_type():
    tu = get_tu('typedef int foo;')
    typedef = get_cursor(tu, 'foo')
    assert typedef is not None

    assert typedef.kind.is_declaration()
    underlying = typedef.underlying_typedef_type
    assert underlying.kind == TypeKind.INT

kParentTest = """\
        class C {
            void f();
        }

        void C::f() { }
    """
def test_semantic_parent():
    tu = get_tu(kParentTest, 'cpp')
    curs = get_cursors(tu, 'f')
    decl = get_cursor(tu, 'C')
    assert(len(curs) == 2)
    assert(curs[0].semantic_parent == curs[1].semantic_parent)
    assert(curs[0].semantic_parent == decl)

def test_lexical_parent():
    tu = get_tu(kParentTest, 'cpp')
    curs = get_cursors(tu, 'f')
    decl = get_cursor(tu, 'C')
    assert(len(curs) == 2)
    assert(curs[0].lexical_parent != curs[1].lexical_parent)
    assert(curs[0].lexical_parent == decl)
    assert(curs[1].lexical_parent == tu.cursor)

def test_enum_type():
    tu = get_tu('enum TEST { FOO=1, BAR=2 };')
    enum = get_cursor(tu, 'TEST')
    assert enum is not None

    assert enum.kind == CursorKind.ENUM_DECL
    enum_type = enum.enum_type
    assert enum_type.kind == TypeKind.UINT

def test_enum_type_cpp():
    tu = get_tu('enum TEST : long long { FOO=1, BAR=2 };', lang="cpp")
    enum = get_cursor(tu, 'TEST')
    assert enum is not None

    assert enum.kind == CursorKind.ENUM_DECL
    assert enum.enum_type.kind == TypeKind.LONGLONG

def test_objc_type_encoding():
    tu = get_tu('int i;', lang='objc')
    i = get_cursor(tu, 'i')

    assert i is not None
    assert i.objc_type_encoding == 'i'

def test_enum_values():
    tu = get_tu('enum TEST { SPAM=1, EGG, HAM = EGG * 20};')
    enum = get_cursor(tu, 'TEST')
    assert enum is not None

    assert enum.kind == CursorKind.ENUM_DECL

    enum_constants = list(enum.get_children())
    assert len(enum_constants) == 3

    spam, egg, ham = enum_constants

    assert spam.kind == CursorKind.ENUM_CONSTANT_DECL
    assert spam.enum_value == 1
    assert egg.kind == CursorKind.ENUM_CONSTANT_DECL
    assert egg.enum_value == 2
    assert ham.kind == CursorKind.ENUM_CONSTANT_DECL
    assert ham.enum_value == 40

def test_enum_values_cpp():
    tu = get_tu('enum TEST : long long { SPAM = -1, HAM = 0x10000000000};', lang="cpp")
    enum = get_cursor(tu, 'TEST')
    assert enum is not None

    assert enum.kind == CursorKind.ENUM_DECL

    enum_constants = list(enum.get_children())
    assert len(enum_constants) == 2

    spam, ham = enum_constants

    assert spam.kind == CursorKind.ENUM_CONSTANT_DECL
    assert spam.enum_value == -1
    assert ham.kind == CursorKind.ENUM_CONSTANT_DECL
    assert ham.enum_value == 0x10000000000

def test_annotation_attribute():
    tu = get_tu('int foo (void) __attribute__ ((annotate("here be annotation attribute")));')

    foo = get_cursor(tu, 'foo')
    assert foo is not None

    for c in foo.get_children():
        if c.kind == CursorKind.ANNOTATE_ATTR:
            assert c.displayname == "here be annotation attribute"
            break
    else:
        assert False, "Couldn't find annotation"

def test_result_type():
    tu = get_tu('int foo();')
    foo = get_cursor(tu, 'foo')

    assert foo is not None
    t = foo.result_type
    assert t.kind == TypeKind.INT

def test_get_tokens():
    """Ensure we can map cursors back to tokens."""
    tu = get_tu('int foo(int i);')
    foo = get_cursor(tu, 'foo')

    tokens = list(foo.get_tokens())
    assert len(tokens) == 6
    assert tokens[0].spelling == 'int'
    assert tokens[1].spelling == 'foo'

def test_get_arguments():
    tu = get_tu('void foo(int i, int j);')
    foo = get_cursor(tu, 'foo')
    arguments = list(foo.get_arguments())

    assert len(arguments) == 2
    assert arguments[0].spelling == "i"
    assert arguments[1].spelling == "j"

kTemplateArgTest = """\
        template <int kInt, typename T, bool kBool>
        void foo();

        template<>
        void foo<-7, float, true>();
    """

def test_get_num_template_arguments():
    tu = get_tu(kTemplateArgTest, lang='cpp')
    foos = get_cursors(tu, 'foo')

    assert foos[1].get_num_template_arguments() == 3

def test_get_template_argument_kind():
    tu = get_tu(kTemplateArgTest, lang='cpp')
    foos = get_cursors(tu, 'foo')

    assert foos[1].get_template_argument_kind(0) == TemplateArgumentKind.INTEGRAL
    assert foos[1].get_template_argument_kind(1) == TemplateArgumentKind.TYPE
    assert foos[1].get_template_argument_kind(2) == TemplateArgumentKind.INTEGRAL

def test_get_template_argument_type():
    tu = get_tu(kTemplateArgTest, lang='cpp')
    foos = get_cursors(tu, 'foo')

    assert foos[1].get_template_argument_type(1).kind == TypeKind.FLOAT

def test_get_template_argument_value():
    tu = get_tu(kTemplateArgTest, lang='cpp')
    foos = get_cursors(tu, 'foo')

    assert foos[1].get_template_argument_value(0) == -7
    assert foos[1].get_template_argument_value(2) == True

def test_get_template_argument_unsigned_value():
    tu = get_tu(kTemplateArgTest, lang='cpp')
    foos = get_cursors(tu, 'foo')

    assert foos[1].get_template_argument_unsigned_value(0) == 2 ** 32 - 7
    assert foos[1].get_template_argument_unsigned_value(2) == True

def test_referenced():
    tu = get_tu('void foo(); void bar() { foo(); }')
    foo = get_cursor(tu, 'foo')
    bar = get_cursor(tu, 'bar')
    for c in bar.get_children():
        if c.kind == CursorKind.CALL_EXPR:
            assert c.referenced.spelling == foo.spelling
            break

def test_mangled_name():
    kInputForMangling = """\
    int foo(int, int);
    """
    tu = get_tu(kInputForMangling, lang='cpp')
    foo = get_cursor(tu, 'foo')

    # Since libclang does not link in targets, we cannot pass a triple to it
    # and force the target. To enable this test to pass on all platforms, accept
    # all valid manglings.
    # [c-index-test handles this by running the source through clang, emitting
    #  an AST file and running libclang on that AST file]
    assert foo.mangled_name in ('_Z3fooii', '__Z3fooii', '?foo@@YAHHH')
