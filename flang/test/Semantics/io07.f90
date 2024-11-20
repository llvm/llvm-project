! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
1001 format(A)

     !ERROR: Format statement must be labeled
     format(A)

2001 format(3I8, 3Z8)
2002 format(DC, F8.2, 3I8, Z8)
2003 format(  D C , F 8 . 2 , 3  I  8  ,  3  Z  8  )
2004 format(20PF10.2)
2005 format(20P,F10.2)
2006 format(20P7F10.2)
2007 format(1X/)
2008 format(/02x)
2009 format(1x/02x)
2010 format(2L2:)
2011 format(:2L2)
2012 format(2L2 : 2L2)

     write(*,2013) 'Hello'
     if (2+2.eq.4) then
2013   format(A10) ! ok to reference outside the if block
     endif

     ! C1302 warnings; no errors
     !WARNING: Expected ',' or ')' in format expression
2051 format(1X3/)
     !WARNING: Expected ',' or ')' in format expression
2052 format(1X003/)
     !WARNING: Expected ',' or ')' in format expression
2053 format(3P7I2)
     !WARNING: Expected ',' or ')' in format expression
2054 format(3PI2)

     !ERROR: Expected ',' or ')' in format expression
2101 format(3I83Z8, 'abc')

     !ERROR: Expected ',' or ')' in format expression
2102 format(  3  I  8  3  Z  8  )

     !ERROR: Expected ',' or ')' in format expression
2103 format(3I8 3Z8)

     !WARNING: Expected ',' or ')' in format expression
2104 format(3I8 Z8)

3001 format(*(I3))
3002 format(5X,*(2(A)))

     !ERROR: Unlimited format item list must contain a data edit descriptor
     !WARNING: 'X' edit descriptor must have a positive position value
3101 format(*(X))

     !ERROR: Unlimited format item list must contain a data edit descriptor
3102 format(5X,*(2(/)))

     !ERROR: Unlimited format item list must contain a data edit descriptor
3103 format(5X, 'abc', *((:)))

     !WARNING: 'X' edit descriptor must have a positive position value
4001 format(2(X))

     !ERROR: List repeat specifier must be positive
     !WARNING: 'X' edit descriptor must have a positive position value
     !ERROR: 'DT' edit descriptor repeat specifier must be positive
4101 format(0(X), 0dt)

6001 format(((I0, B0)))

     !ERROR: 'A' edit descriptor 'w' value must be positive
     !ERROR: 'L' edit descriptor 'w' value must be positive
6101 format((A0), ((L0)))

     !ERROR: 'L' edit descriptor 'w' value must be positive
6102 format((3(((L 0 0 0)))))

7001 format(17G8.1, 17G8.1e3)

     !ERROR: Expected 'G' edit descriptor '.d' value
7101 format(17G8)

8001 format(9G0.5)

     !ERROR: A 'G0' edit descriptor must not have an 'e' value
8101 format(9(G0.5e1))

     !ERROR: A 'G0' edit descriptor must not have an 'e' value
8102 format(9(G0.5  E 1))
end
