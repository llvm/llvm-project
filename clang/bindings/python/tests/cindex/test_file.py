from clang.cindex import Index, File

def test_file():
  index = Index.create()
  tu = index.parse('t.c', unsaved_files = [('t.c', "")])
  file = File.from_name(tu, "t.c")
  assert str(file) == "t.c"
  assert file.name == "t.c"
  assert repr(file) == "<File: t.c>"
