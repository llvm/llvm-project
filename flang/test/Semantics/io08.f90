! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
  write(*,*)
  write(*,'()')
  write(*,'(A)')
  write(*,'(2X:2X)')
  write(*,'(2X/2X)')
  write(*,'(3/2X)')
  write(*,'(3PF5.2)')
  write(*,'(+3PF5.2)')
  write(*,'(-3PF5.2)')
  write(*,'(000p,10p,0p)')
  write(*,'(3P7D5.2)')
  write(*,'(3P,7F5.2)')
  write(*,'(2X,(i3))')
  write(*,'(5X,*(2X,I2))')
  write(*,'(5X,*(2X,DT))')
  write(*,'(*(DT))')
  write(*,'(*(DT"value"))')
  write(*,'(*(DT(+1,0,-1)))')
  write(*,'(*(DT"value"(+1,000,-1)))')
  write(*,'(*(DT(0)))')
  write(*,'(S,(RZ),2E10.3)')
  write(*,'(7I2)')
  write(*,'(07I02)')
  write(*,'(07I02.01)')
  write(*,'(07I02.02)')
  write(*,'(I0)')
  write(*,'(G4.2)')
  write(*,'(G0.8)')
  write(*,'(T3)')
  write(*,'("abc")')
  write(*,'("""abc""")')
  write(*,'("a""""bc", 2x)')
  write(*,'(3Habc)')
  write(*,'(3Habc, 2X, 3X)')
  write(*,'(987654321098765432X)')
  write(*,'($)')
  write(*,'(\)')
  write(*,'(RZ,RU,RP,RN,RD,RC,SS,SP,S,3G15.3e2)')

  ! C1302 warnings; no errors
  write(*,'(3P7I2)')
  write(*,'(5X i3)')
  write(*,'(XEN)')

  !ERROR: Empty format expression
  write(*,"")

  !ERROR: Empty format expression
  write(*,"" // '' // "")

  !ERROR: Format expression must have an initial '('
  write(*,'I3')

  !ERROR: Unexpected '+' in format expression
  write(*,'(+7I2)')

  !ERROR: Unexpected '-' in format expression
  write(*,'(-7I2)')

  !ERROR: 'P' edit descriptor must have a scale factor
  write(*,'(P7F5.2)')

  !ERROR: 'P' edit descriptor must have a scale factor
  write(*,'(P7F' // '5.2)')

  !ERROR: Unexpected integer constant
  write(*,'(X,3,3L4)')

  !ERROR: Unexpected ',' before ')' in format expression
  write(*,'(X,i3,)')

  !ERROR: Unexpected ',' in format expression
  write(*,'(X,i3,,)')

  !ERROR: Unexpected ',' in format expression
  !ERROR: Unexpected ',' before ')' in format expression
  write(*,'(X,i3,,,)')

  !ERROR: Unexpected ',' before ')' in format expression
  write(*,'(X,(i3,))')

  !ERROR: Unexpected '*' in format expression
  write(*,'(*)')

  !ERROR: Expected integer constant in 'DT' edit descriptor v-list
  write(*,'(*(DT(+1,0,=1)))')

  !ERROR: Expected integer constant in 'DT' edit descriptor v-list
  write(*,'(DT(1,0,+))')

  !ERROR: Expected integer constant in 'DT' edit descriptor v-list
  write(*,'(DT(1,0,*))')

  !ERROR: Expected ',' or ')' in 'DT' edit descriptor v-list
  write(*,'(DT(1,0,2*))')

  !ERROR: Expected ',' or ')' in 'DT' edit descriptor v-list
  write(*,'(DT(1,0,2*,+,?))')

  !ERROR: Expected integer constant in 'DT' edit descriptor v-list
  !ERROR: Unterminated format expression
  write(*,'(DT(1,0,*)')

  !ERROR: Expected ',' or ')' in 'DT' edit descriptor v-list
  !ERROR: Unterminated format expression
  write(*,'(DT(1,0,2*,+,?)')

  !ERROR: Unexpected '?' in format expression
  !ERROR: Unexpected ',' in format expression
  write(*,'(?,*(DT(+1,,1)))')

  !ERROR: Repeat specifier before unlimited format item list
  !ERROR: Unlimited format item list must contain a data edit descriptor
   write(*,'(5X,3*(2(X)))')

  !ERROR: Nested unlimited format item list
  write(*,'(D12.2,(*(F10.2)))')

  !ERROR: Unlimited format item list must contain a data edit descriptor
  write(*,'(5X,*(2(X)))')

  !ERROR: Character in format after unlimited format item list
  write(*,'(*(Z5),*(2F20.3))')

  !ERROR: Character in format after unlimited format item list
  write(*,'(*(B5),*(2(I5)))')

  !ERROR: Character in format after unlimited format item list
  write(*,'(*(I5), D12.7)')

  !ERROR: 'I' edit descriptor 'm' value is greater than 'w' value
  write(*,'(07I02.0 3)')

  !ERROR: 'Z' edit descriptor 'm' value is greater than 'w' value
  write(*,'(07Z02.4)')

  !ERROR: 'I' edit descriptor repeat specifier must be positive
  write(*,'(0I2)')

  !ERROR: List repeat specifier must be positive
  write(*,'(0(I2))')

  !ERROR: List repeat specifier must be positive
  write(*,'(000(I2))')

  !ERROR: List repeat specifier must be positive
  !ERROR: 'I' edit descriptor repeat specifier must be positive
  write(*,'(0(0I2))')

  !ERROR: Kind parameter '_' character in format expression
  write(*,'(5_4X)')

  !ERROR: Unexpected '+' in format expression
  write(*,'(I+3)')

  !ERROR: Unexpected '-' in format expression
  write(*,'(I-3)')

  !ERROR: Unexpected '-' in format expression
  write(*,'(I-3, X)')

  !ERROR: 'X' edit descriptor must have a positive position value
  write(*,'(0X)')

  !ERROR: Unexpected 'Y' in format expression
  write(*,'(XY)')

  !ERROR: Unexpected 'Y' in format expression
  write(*,'(XYM)')

  !ERROR: Unexpected 'M' in format expression
  write(*,'(MXY)')

  !ERROR: Unexpected 'R' in format expression
  !ERROR: Unexpected 'R' in format expression
  write(*,"(RR, RV)")

  !ERROR: Unexpected '-' in format expression
  !ERROR: Unexpected 'Y' in format expression
  write(*,'(I-3, XY)')

  !ERROR: 'A' edit descriptor 'w' value must be positive
  write(*,'(A0)')

  !ERROR: 'L' edit descriptor 'w' value must be positive
  write(*,'(L0)')

  !ERROR: Expected 'G' edit descriptor '.d' value
  write(*,'(G4)')

  !ERROR: Unexpected 'e' in 'G0' edit descriptor
  write(*,'(G0.8e)')

  !ERROR: Unexpected 'e' in 'G0' edit descriptor
  write(*,'(G0.8e2)')

  !ERROR: Kind parameter '_' character in format expression
  write(*,'(I5_4)')

  !ERROR: Kind parameter '_' character in format expression
  write(*,'(5_4P)')

  !ERROR: 'T' edit descriptor must have a positive position value
  write(*,'(T0)')

  !ERROR: 'T' edit descriptor must have a positive position value
  !ERROR: Unterminated format expression
  write(*,'(T0')

  !ERROR: 'TL' edit descriptor must have a positive position value
  !ERROR: 'T' edit descriptor must have a positive position value
  !ERROR: Expected 'EN' edit descriptor 'd' value after '.'
  write(*,'(TL0,T0,EN12.)')

  !ERROR: Expected 'EX' edit descriptor 'e' value after 'E'
  write(*,'(EX12.3e2, EX12.3e)')

  !ERROR: 'TL' edit descriptor must have a positive position value
  !ERROR: 'T' edit descriptor must have a positive position value
  !ERROR: Unterminated format expression
  write(*,'(TL00,T000')

  !ERROR: Unterminated format expression
  write(*,'(')

  !ERROR: Unterminated format expression
  write(*,'(-')

  !ERROR: Unterminated format expression
  write(*,'(I3+')

  !ERROR: Unterminated format expression
  write(*,'(I3,-')

  !ERROR: Unexpected integer constant
  write(*,'(3)')

  !ERROR: Unexpected ',' before ')' in format expression
  write(*,'(3,)')

  !ERROR: Unexpected ',' in format expression
  write(*,'(,3)')

  !ERROR: Unexpected ',' before ')' in format expression
  write(*,'(,)')

  !ERROR: Unterminated format expression
  write(*,'(X')

  !ERROR: Unterminated format expression
  write(*,'(XX') ! C1302 warning is not an error

  !ERROR: Unexpected '@' in format expression
  !ERROR: Unexpected '#' in format expression
  !ERROR: Unexpected '&' in format expression
  write(*,'(@@, #  ,&&& &&, ignore error 4)')

  !ERROR: Repeat specifier before 'TR' edit descriptor
  write(*,'(3TR0)')

  !ERROR: 'TR' edit descriptor must have a positive position value
  write(*,'(TR0)')

  !ERROR: Kind parameter '_' character in format expression
  write(*,'(3_4X)')

  !ERROR: Kind parameter '_' character in format expression
  write(*,'(1_"abc")')

  !ERROR: Unterminated string
  !ERROR: Unterminated format expression
  write(*,'("abc)')

  !ERROR: Unexpected '_' in format expression
  write(*,'("abc"_1)')

  !ERROR: Unexpected '@' in format expression
  write(*,'(3Habc, 3@, X)')

  !ERROR: Unterminated format expression
  write(*,'(4Habc)')

  !ERROR: Unterminated 'H' edit descriptor
  !ERROR: Unterminated format expression
  write(*,'(5Habc)')

  !ERROR: Unterminated 'H' edit descriptor
  !ERROR: Unterminated format expression
  write(*,'(50Habc)')

  !ERROR: Integer overflow in format expression
  write(*,'(9876543210987654321X)')

  !ERROR: Integer overflow in format expression
  write(*,'(98765432109876543210X)')

  !ERROR: Integer overflow in format expression
  write(*,'(I98765432109876543210)')

  !ERROR: Integer overflow in format expression
  write(*,'(45I20.98765432109876543210, 45I20)')

  !ERROR: Integer overflow in format expression
  write(*,'(45' // '  I20.9876543' // '2109876543210, 45I20)')

  !ERROR: Repeat specifier before '$' edit descriptor
  write(*,'(7$)')
end
