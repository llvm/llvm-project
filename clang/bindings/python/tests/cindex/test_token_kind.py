from clang.cindex import TokenKind
from nose.tools import eq_
from nose.tools import ok_
from nose.tools import raises

def test_constructor():
    """Ensure TokenKind constructor works as expected."""

    t = TokenKind(5, 'foo')

    eq_(t.value, 5)
    eq_(t.name, 'foo')

@raises(ValueError)
def test_bad_register():
    """Ensure a duplicate value is rejected for registration."""

    TokenKind.register(2, 'foo')

@raises(ValueError)
def test_unknown_value():
    """Ensure trying to fetch an unknown value raises."""

    TokenKind.from_value(-1)

def test_registration():
    """Ensure that items registered appear as class attributes."""
    ok_(hasattr(TokenKind, 'LITERAL'))
    literal = TokenKind.LITERAL

    ok_(isinstance(literal, TokenKind))

def test_from_value():
    """Ensure registered values can be obtained from from_value()."""
    t = TokenKind.from_value(3)
    ok_(isinstance(t, TokenKind))
    eq_(t, TokenKind.LITERAL)

def test_repr():
    """Ensure repr() works."""

    r = repr(TokenKind.LITERAL)
    eq_(r, 'TokenKind.LITERAL')
