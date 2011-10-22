from clang.cindex import Index, CursorKind, TypeKind

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
    index = Index.create()
    tu = index.parse('t.c', unsaved_files = [('t.c',kInput)])

    for n in tu.cursor.get_children():
        if n.spelling == 'teststruct':
            fields = list(n.get_children())

            assert all(x.kind == CursorKind.FIELD_DECL for x in fields)

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

            break

    else:
        assert False, "Didn't find teststruct??"


constarrayInput="""
struct teststruct {
  void *A[2];
};
"""
def testConstantArray():
    index = Index.create()
    tu = index.parse('t.c', unsaved_files = [('t.c',constarrayInput)])

    for n in tu.cursor.get_children():
        if n.spelling == 'teststruct':
            fields = list(n.get_children())
            assert fields[0].spelling == 'A'
            assert fields[0].type.kind == TypeKind.CONSTANTARRAY
            break
    else:
        assert False, "Didn't find teststruct??"
