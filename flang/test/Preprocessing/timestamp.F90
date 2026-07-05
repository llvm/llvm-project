!RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
!CHECK: INTEGER, PARAMETER :: tslen = 24_4
!CHECK: LOGICAL, PARAMETER :: tsspaces = .true._4
!CHECK: LOGICAL, PARAMETER :: tscolons = .true._4

integer, parameter :: tsLen = len(__TIMESTAMP__)
character(tsLen), parameter :: ts = __TIMESTAMP__
integer, parameter :: spaces(*) = [4, 8, 11, 20]
integer, parameter :: colons(*) = [14, 17]
logical, parameter :: tsSpaces = all([character(1)::(ts(spaces(j):spaces(j)),j=1,size(spaces))] == ' ')
logical, parameter :: tsColons = all([character(1)::(ts(colons(j):colons(j)),j=1,size(colons))] == ':')
end
