! RUN: %flang_fc1 -fdebug-unparse-no-sema %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 | FileCheck %s -check-prefix=TREE

! Test parsing of F2023 enumeration type definitions (R766-R769)

module m
  ! Basic enumeration type with double-colon
  ! CHECK: ENUMERATION TYPE :: color
  ! CHECK: ENUMERATOR :: red, green, blue
  ! CHECK: END ENUMERATION TYPE color
  ! TREE: EnumerationTypeDef
  ! TREE: EnumerationTypeStmt
  ! TREE: Name = 'color'
  ! TREE: EnumerationEnumeratorStmt
  ! TREE: Name = 'red'
  ! TREE: Name = 'green'
  ! TREE: Name = 'blue'
  ! TREE: EndEnumerationTypeStmt
  ! TREE: Name = 'color'
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type color

  ! Without double-colon on ENUMERATION TYPE statement
  ! CHECK: ENUMERATION TYPE :: direction
  ! CHECK: ENUMERATOR :: north, south, east, west
  ! CHECK: END ENUMERATION TYPE direction
  ! TREE: EnumerationTypeDef
  ! TREE: EnumerationTypeStmt
  ! TREE: Name = 'direction'
  ! TREE: EnumerationEnumeratorStmt
  ! TREE: Name = 'north'
  ! TREE: Name = 'south'
  ! TREE: Name = 'east'
  ! TREE: Name = 'west'
  ! TREE: EndEnumerationTypeStmt
  ! TREE: Name = 'direction'
  enumeration type direction
    enumerator north, south, east, west
  end enumeration type direction

  ! With access-spec (PUBLIC)
  ! CHECK: ENUMERATION TYPE, PUBLIC :: priority
  ! CHECK: ENUMERATOR :: low, medium, high
  ! CHECK: END ENUMERATION TYPE priority
  ! TREE: EnumerationTypeDef
  ! TREE: EnumerationTypeStmt
  ! TREE: AccessSpec -> Kind = Public
  ! TREE: Name = 'priority'
  ! TREE: EnumerationEnumeratorStmt
  ! TREE: Name = 'low'
  ! TREE: Name = 'medium'
  ! TREE: Name = 'high'
  ! TREE: EndEnumerationTypeStmt
  enumeration type, public :: priority
    enumerator :: low, medium, high
  end enumeration type priority

  ! With access-spec (PRIVATE)
  ! CHECK: ENUMERATION TYPE, PRIVATE :: internal_state
  ! CHECK: ENUMERATOR :: idle, running
  ! CHECK: END ENUMERATION TYPE internal_state
  ! TREE: EnumerationTypeDef
  ! TREE: EnumerationTypeStmt
  ! TREE: AccessSpec -> Kind = Private
  ! TREE: Name = 'internal_state'
  ! TREE: EnumerationEnumeratorStmt
  ! TREE: Name = 'idle'
  ! TREE: Name = 'running'
  ! TREE: EndEnumerationTypeStmt
  enumeration type, private :: internal_state
    enumerator :: idle, running
  end enumeration type internal_state

  ! Multiple ENUMERATOR statements
  ! CHECK: ENUMERATION TYPE :: season
  ! CHECK: ENUMERATOR :: spring
  ! CHECK: ENUMERATOR :: summer
  ! CHECK: ENUMERATOR :: autumn, winter
  ! CHECK: END ENUMERATION TYPE season
  ! TREE: EnumerationTypeDef
  ! TREE: EnumerationTypeStmt
  ! TREE: Name = 'season'
  ! TREE: EnumerationEnumeratorStmt
  ! TREE: Name = 'spring'
  ! TREE: EnumerationEnumeratorStmt
  ! TREE: Name = 'summer'
  ! TREE: EnumerationEnumeratorStmt
  ! TREE: Name = 'autumn'
  ! TREE: Name = 'winter'
  ! TREE: EndEnumerationTypeStmt
  ! TREE: Name = 'season'
  enumeration type :: season
    enumerator :: spring
    enumerator :: summer
    enumerator :: autumn, winter
  end enumeration type season

  ! End statement without name
  ! CHECK: ENUMERATION TYPE :: simple
  ! CHECK: ENUMERATOR :: a, b
  ! CHECK: END ENUMERATION TYPE
  ! TREE: EnumerationTypeDef
  ! TREE: EnumerationTypeStmt
  ! TREE: Name = 'simple'
  ! TREE: EnumerationEnumeratorStmt
  ! TREE: Name = 'a'
  ! TREE: Name = 'b'
  ! TREE: EndEnumerationTypeStmt
  enumeration type :: simple
    enumerator :: a, b
  end enumeration type
end module
