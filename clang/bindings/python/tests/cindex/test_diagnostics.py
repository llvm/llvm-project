from clang.cindex import *
from .util import get_tu

# FIXME: We need support for invalid translation units to test better.

def test_diagnostic_warning():
    tu = get_tu('int f0() {}\n')
    assert len(tu.diagnostics) == 1
    assert tu.diagnostics[0].severity == Diagnostic.Warning
    assert tu.diagnostics[0].location.line == 1
    assert tu.diagnostics[0].location.column == 11
    assert (tu.diagnostics[0].spelling ==
            'control reaches end of non-void function')

def test_diagnostic_note():
    # FIXME: We aren't getting notes here for some reason.
    tu = get_tu('#define A x\nvoid *A = 1;\n')
    assert len(tu.diagnostics) == 1
    assert tu.diagnostics[0].severity == Diagnostic.Warning
    assert tu.diagnostics[0].location.line == 2
    assert tu.diagnostics[0].location.column == 7
    assert 'incompatible' in tu.diagnostics[0].spelling
#    assert tu.diagnostics[1].severity == Diagnostic.Note
#    assert tu.diagnostics[1].location.line == 1
#    assert tu.diagnostics[1].location.column == 11
#    assert tu.diagnostics[1].spelling == 'instantiated from'

def test_diagnostic_fixit():
    tu = get_tu('struct { int f0; } x = { f0 : 1 };')
    assert len(tu.diagnostics) == 1
    assert tu.diagnostics[0].severity == Diagnostic.Warning
    assert tu.diagnostics[0].location.line == 1
    assert tu.diagnostics[0].location.column == 26
    assert tu.diagnostics[0].spelling.startswith('use of GNU old-style')
    assert len(tu.diagnostics[0].fixits) == 1
    assert tu.diagnostics[0].fixits[0].range.start.line == 1
    assert tu.diagnostics[0].fixits[0].range.start.column == 26
    assert tu.diagnostics[0].fixits[0].range.end.line == 1
    assert tu.diagnostics[0].fixits[0].range.end.column == 30
    assert tu.diagnostics[0].fixits[0].value == '.f0 = '

def test_diagnostic_range():
    tu = get_tu('void f() { int i = "a" + 1; }')
    assert len(tu.diagnostics) == 1
    assert tu.diagnostics[0].severity == Diagnostic.Warning
    assert tu.diagnostics[0].location.line == 1
    assert tu.diagnostics[0].location.column == 16
    assert tu.diagnostics[0].spelling.startswith('incompatible pointer to')
    assert len(tu.diagnostics[0].fixits) == 0
    assert len(tu.diagnostics[0].ranges) == 1
    assert tu.diagnostics[0].ranges[0].start.line == 1
    assert tu.diagnostics[0].ranges[0].start.column == 20
    assert tu.diagnostics[0].ranges[0].end.line == 1
    assert tu.diagnostics[0].ranges[0].end.column == 27
    try:
      tu.diagnostics[0].ranges[1].start.line
    except IndexError:
      assert True
    else:
      assert False

def test_diagnostic_category():
    """Ensure that category properties work."""
    tu = get_tu('int f(int i) { return 7; }', all_warnings=True)
    assert len(tu.diagnostics) == 1
    d = tu.diagnostics[0]

    assert d.severity == Diagnostic.Warning
    assert d.location.line == 1
    assert d.location.column == 11

    assert d.category_number == 2
    assert d.category_name == 'Semantic Issue'

def test_diagnostic_option():
    """Ensure that category option properties work."""
    tu = get_tu('int f(int i) { return 7; }', all_warnings=True)
    assert len(tu.diagnostics) == 1
    d = tu.diagnostics[0]

    assert d.option == '-Wunused-parameter'
    assert d.disable_option == '-Wno-unused-parameter'

def test_diagnostic_children():
    tu = get_tu('void f(int x) {} void g() { f(); }')
    assert len(tu.diagnostics) == 1
    d = tu.diagnostics[0]

    children = d.children
    assert len(children) == 1
    assert children[0].severity == Diagnostic.Note
    assert children[0].spelling.endswith('declared here')
    assert children[0].location.line == 1
    assert children[0].location.column == 1

def test_diagnostic_string_repr():
    tu = get_tu('struct MissingSemicolon{}')
    assert len(tu.diagnostics) == 1
    d = tu.diagnostics[0]

    assert repr(d) == '<Diagnostic severity 3, location <SourceLocation file \'t.c\', line 1, column 26>, spelling "expected \';\' after struct">'
    
