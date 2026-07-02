! RUN: %check_flang_tidy %s bugprone-subprogram-trampoline %t
program trampoline_test
  call process(inner)  ! This will trigger a warning
  ! CHECK-MESSAGES: :[[@LINE-1]]:3: warning: contained subprogram 'inner' is passed as an argument
contains
  subroutine inner()
    print *, "Inside inner"
  end subroutine

  subroutine process(proc)
    interface
      subroutine proc()
      end subroutine
    end interface
    call proc()
  end subroutine
end program trampoline_test
