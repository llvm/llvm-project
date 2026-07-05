! RUN: %flang_fc1 -E -fpreprocess-include-lines -I %S/Inputs %s 2>&1 | FileCheck %s
      include 'include-file'
      include "include-file"
      include 1_'include-file'
      include 1_"include-file"
      i n c l u d e 'include-file'
      INCLUDE 'include-file'
      I N C L U D E 'include-file'
include 'include-file'
include "include-file"
include 1_'include-file'
include 1_"include-file"
i n c l u d e 'include-file'
INCLUDE 'include-file'
I N C L U D E 'include-file'
     0include 'include-file'
      x = 2
     include 'include-file'
      print *, "
     1include 'not-an-include'
     2"
cinclude 'not-an-include'
*include 'not-an-include'
!include 'not-an-include'
c     include 'not-an-include'
*     include 'not-an-include'
!     include 'not-an-include'
      end

!CHECK:#line "{{.*[/\\]}}include-file" 1
!CHECK:      x = 1
!CHECK:#line "{{.*[/\\]}}include-file" 1
!CHECK:      x = 1
!CHECK:#line "{{.*[/\\]}}include-file" 1
!CHECK:      x = 1
!CHECK:#line "{{.*[/\\]}}include-file" 1
!CHECK:      x = 1
!CHECK:#line "{{.*[/\\]}}include-file" 1
!CHECK:      x = 1
!CHECK:#line "{{.*[/\\]}}include-file" 1
!CHECK:      x = 1
!CHECK:#line "{{.*[/\\]}}include-file" 1
!CHECK:      x = 1
!CHECK:#line "{{.*[/\\]}}include-file" 1
!CHECK:      x = 1
!CHECK:#line "{{.*[/\\]}}include-file" 1
!CHECK:      x = 1
!CHECK:#line "{{.*[/\\]}}include-file" 1
!CHECK:      x = 1
!CHECK:#line "{{.*[/\\]}}include-file" 1
!CHECK:      x = 1
!CHECK:#line "{{.*[/\\]}}include-file" 1
!CHECK:      x = 1
!CHECK:#line "{{.*[/\\]}}include-file" 1
!CHECK:      x = 1
!CHECK:#line "{{.*[/\\]}}include-file" 1
!CHECK:      x = 1
!CHECK:#line "{{.*[/\\]}}include-file" 1
!CHECK:      x = 1
!CHECK:#line "{{.*[/\\]}}include.f" 17
!CHECK:      x = 2
!CHECK:#line "{{.*[/\\]}}include-file" 1
!CHECK:      x = 1
!CHECK:#line "{{.*[/\\]}}include.f" 19
!CHECK:      print *, "                                                        &
!CHECK:     &include 'not-an-include'                                          &
!CHECK:     &"
!CHECK:      end
