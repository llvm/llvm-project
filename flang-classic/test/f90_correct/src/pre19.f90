!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! This tests stringification.  The result should be: 
!   Match:rain
!   Match:snow
!

#define doCompare(NAME) if(cmp(a%NAME,b%NAME,sizeof(a%NAME))) print *, "Match:", #NAME

module weather
    type WeatherType
        real rain
        real snow
    end type
   
    contains

    ! Dummy routine, just for syntax and macro testing purposes
    function cmp(a, b, sz)
        real :: a, b
        integer :: sz ! Ignored
        cmp = a .eq. b

        if (cmp .eq. .true.) then
            call check(.true., .true., 1)
        else
            call check(.false., .true., 1)
        endif
    end function cmp

    subroutine compareWeather(a,b)
        type (WeatherType) :: a,b
        doCompare(rain)
        doCompare(snow)
    end subroutine compareWeather
end module weather

program p
    use weather
    logical :: res(1) = .false., expect(1) = .true.
    type(WeatherType) :: foo = WeatherType(1.0, 2.0)
    type(WeatherType) :: bar = WeatherType(1.0, 2.0)
    call compareWeather(foo, bar)
end program
