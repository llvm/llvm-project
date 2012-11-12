from clang.cindex import CursorKind
from clang.cindex import Index
from clang.cindex import SourceLocation
from clang.cindex import SourceRange
from clang.cindex import TokenKind
from nose.tools import eq_
from nose.tools import ok_

from .util import get_tu

def test_token_to_cursor():
    """Ensure we can obtain a Cursor from a Token instance."""
    tu = get_tu('int i = 5;')
    r = tu.get_extent('t.c', (0, 9))
    tokens = list(tu.get_tokens(extent=r))

    assert len(tokens) == 5
    assert tokens[1].spelling == 'i'
    assert tokens[1].kind == TokenKind.IDENTIFIER

    cursor = tokens[1].cursor
    assert cursor.kind == CursorKind.VAR_DECL
    assert tokens[1].cursor == tokens[2].cursor

def test_token_location():
    """Ensure Token.location works."""

    tu = get_tu('int foo = 10;')
    r = tu.get_extent('t.c', (0, 11))

    tokens = list(tu.get_tokens(extent=r))
    eq_(len(tokens), 4)

    loc = tokens[1].location
    ok_(isinstance(loc, SourceLocation))
    eq_(loc.line, 1)
    eq_(loc.column, 5)
    eq_(loc.offset, 4)

def test_token_extent():
    """Ensure Token.extent works."""
    tu = get_tu('int foo = 10;')
    r = tu.get_extent('t.c', (0, 11))

    tokens = list(tu.get_tokens(extent=r))
    eq_(len(tokens), 4)

    extent = tokens[1].extent
    ok_(isinstance(extent, SourceRange))

    eq_(extent.start.offset, 4)
    eq_(extent.end.offset, 7)
