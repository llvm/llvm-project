from clang.cindex import Cursor
from clang.cindex import File
from clang.cindex import SourceLocation
from clang.cindex import SourceRange
from .util import get_cursor
from .util import get_tu

baseInput="int one;\nint two;\n"

def assert_location(loc, line, column, offset):
    assert loc.line == line
    assert loc.column == column
    assert loc.offset == offset

def test_location():
    tu = get_tu(baseInput)
    one = get_cursor(tu, 'one')
    two = get_cursor(tu, 'two')

    assert one is not None
    assert two is not None

    assert_location(one.location,line=1,column=5,offset=4)
    assert_location(two.location,line=2,column=5,offset=13)

    # adding a linebreak at top should keep columns same
    tu = get_tu('\n' + baseInput)
    one = get_cursor(tu, 'one')
    two = get_cursor(tu, 'two')

    assert one is not None
    assert two is not None

    assert_location(one.location,line=2,column=5,offset=5)
    assert_location(two.location,line=3,column=5,offset=14)

    # adding a space should affect column on first line only
    tu = get_tu(' ' + baseInput)
    one = get_cursor(tu, 'one')
    two = get_cursor(tu, 'two')

    assert_location(one.location,line=1,column=6,offset=5)
    assert_location(two.location,line=2,column=5,offset=14)

    # define the expected location ourselves and see if it matches
    # the returned location
    tu = get_tu(baseInput)

    file = File.from_name(tu, 't.c')
    location = SourceLocation.from_position(tu, file, 1, 5)
    cursor = Cursor.from_location(tu, location)

    one = get_cursor(tu, 'one')
    assert one is not None
    assert one == cursor

    # Ensure locations referring to the same entity are equivalent.
    location2 = SourceLocation.from_position(tu, file, 1, 5)
    assert location == location2
    location3 = SourceLocation.from_position(tu, file, 1, 4)
    assert location2 != location3

    offset_location = SourceLocation.from_offset(tu, file, 5)
    cursor = Cursor.from_location(tu, offset_location)
    verified = False
    for n in [n for n in tu.cursor.get_children() if n.spelling == 'one']:
        assert n == cursor
        verified = True

    assert verified

def test_extent():
    tu = get_tu(baseInput)
    one = get_cursor(tu, 'one')
    two = get_cursor(tu, 'two')

    assert_location(one.extent.start,line=1,column=1,offset=0)
    assert_location(one.extent.end,line=1,column=8,offset=7)
    assert baseInput[one.extent.start.offset:one.extent.end.offset] == "int one"

    assert_location(two.extent.start,line=2,column=1,offset=9)
    assert_location(two.extent.end,line=2,column=8,offset=16)
    assert baseInput[two.extent.start.offset:two.extent.end.offset] == "int two"

    file = File.from_name(tu, 't.c')
    location1 = SourceLocation.from_position(tu, file, 1, 1)
    location2 = SourceLocation.from_position(tu, file, 1, 8)

    range1 = SourceRange.from_locations(location1, location2)
    range2 = SourceRange.from_locations(location1, location2)
    assert range1 == range2

    location3 = SourceLocation.from_position(tu, file, 1, 6)
    range3 = SourceRange.from_locations(location1, location3)
    assert range1 != range3
