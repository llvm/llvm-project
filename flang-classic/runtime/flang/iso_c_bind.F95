! 
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
! 

!          THIS CODE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT
!   WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT
!   NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR
!   FITNESS FOR A PARTICULAR PURPOSE.
!


! iso_c_bind.f90
! 32/64 bit  linux and windows.  Add further targets as required.

        module ISO_C_BINDING
#ifdef PGDLL
!DEC$ ATTRIBUTES DLLEXPORT :: ISO_C_BINDING
#endif

        public
        integer C_INT
        parameter ( C_INT = 4)
        integer C_SHORT
        parameter ( C_SHORT = 2 )

#if defined(TARGET_WIN_X8664) 
        integer C_LONG
        parameter ( C_LONG = 4 )
        integer C_INTPTR_T
        parameter ( C_INTPTR_T = 8 )
        integer C_SIZE_T
        parameter ( C_SIZE_T = 8  )
#else
        integer C_LONG
        parameter ( C_LONG = 8 )
        integer C_INTPTR_T
        parameter ( C_INTPTR_T = 8 )
        integer C_SIZE_T
        parameter ( C_SIZE_T = 8  )
#endif

        type C_PTR
        private
        integer (C_INTPTR_T) val
        end type C_PTR

        type C_FUNPTR
        private
        integer (C_INTPTR_T) val
        end type C_FUNPTR

        integer C_LONG_LONG
        parameter ( C_LONG_LONG =  8 )
        integer C_SIGNED_CHAR
        parameter ( C_SIGNED_CHAR = 1 )

! stdint.h
        integer C_INT8_T
        parameter ( C_INT8_T = 1 )
        integer C_INT16_T
        parameter ( C_INT16_T = 2 )
        integer C_INT32_T
        parameter ( C_INT32_T = 4 )
        integer C_INT64_T
        parameter ( C_INT64_T = 8 )
        integer C_INT_LEAST8_T
        parameter ( C_INT_LEAST8_T = 1)
        integer C_INT_LEAST16_T
        parameter ( C_INT_LEAST16_T = 2 )
        integer C_INT_LEAST32_T
        parameter ( C_INT_LEAST32_T = 4 )
        integer C_INT_LEAST64_T
        parameter ( C_INT_LEAST64_T = 8 )
        integer C_INT_FAST8_T
        parameter ( C_INT_FAST8_T =  1)
        integer C_INT_FAST16_T
        parameter ( C_INT_FAST16_T = 8 )
        integer  C_INT_FAST32_T
        parameter ( C_INT_FAST32_T = 8 )
        integer C_INT_FAST64_T
        parameter ( C_INT_FAST64_T = 8 )
        integer C_INTMAX_T
        parameter ( C_INTMAX_T = 8)
        integer C_FLOAT
        parameter ( C_FLOAT = 4 )
        integer C_DOUBLE
        parameter ( C_DOUBLE = 8 )
        integer C_LONG_DOUBLE
#ifdef TARGET_SUPPORTS_QUADFP
        parameter ( C_LONG_DOUBLE = 16  )
#else
        parameter ( C_LONG_DOUBLE = 8  )
#endif
        integer C_FLOAT_COMPLEX
        parameter ( C_FLOAT_COMPLEX = 4 )
        integer C_DOUBLE_COMPLEX
        parameter ( C_DOUBLE_COMPLEX = 8 )
        integer C_LONG_DOUBLE_COMPLEX
#ifdef TARGET_SUPPORTS_QUADFP
        parameter ( C_LONG_DOUBLE_COMPLEX = 16  )
#else
        parameter ( C_LONG_DOUBLE_COMPLEX = 8  )
#endif
        integer C_BOOL
        parameter ( C_BOOL =  1 )
        integer C_CHAR
         parameter ( C_CHAR = 1 )
! C character values
        character(1) C_NULL_CHAR
        parameter (  C_NULL_CHAR = '\0' )

        character(1) C_ALERT 
        parameter (  C_ALERT = '\a')

        character(1) C_BACKSPACE
        parameter (  C_BACKSPACE = '\b')

        character(1) C_FORM_FEED
        parameter (  C_FORM_FEED = '\f')

        character(1) C_NEW_LINE
        parameter (  C_NEW_LINE = '\n')

        character(1) C_CARRIAGE_RETURN
        parameter (  C_CARRIAGE_RETURN = '\r')

        character(1) C_HORIZONTAL_TAB
        parameter (  C_HORIZONTAL_TAB = '\t')

        character(1) C_VERTICAL_TAB
        parameter (  C_VERTICAL_TAB =  '\v')

        type (C_PTR),    parameter :: C_NULL_PTR    = C_PTR(0)
        type (C_FUNPTR), parameter :: C_NULL_FUNPTR = C_FUNPTR(0)

! 04/2009 are adding some code, which will require the .o file gets linked in
        interface operator (.eq.)
            module procedure compare_eq_cptrs
        end interface

        interface operator (.ne.)
            module procedure compare_ne_cptrs
        end interface

        interface operator (.eq.)
            module procedure compare_eq_cfunptrs
        end interface

        interface operator (.ne.)
            module procedure compare_ne_cfunptrs
        end interface

        contains
            logical function compare_eq_cptrs(a,b)
            type(C_PTR) :: a,b
            compare_eq_cptrs = (a%val .eq. b%val)
            return
            end function

            logical function compare_ne_cptrs(a,b)
            type(C_PTR) :: a,b
            compare_ne_cptrs = (a%val .ne. b%val)
            return
            end function

            logical function compare_eq_cfunptrs(a,b)
            type(C_FUNPTR) :: a,b
            compare_eq_cfunptrs = (a%val .eq. b%val)
            return
            end function

            logical function compare_ne_cfunptrs(a,b)
            type(C_FUNPTR) :: a,b
            compare_ne_cfunptrs = (a%val .ne. b%val)
            return
            end function

        end module  ISO_C_BINDING
