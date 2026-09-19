 ! RUN: %check_flang_tidy %s bugprone-mismatched-intent %t
module matrix

  interface
     subroutine inverse(src, dst)
       real, intent(in) :: src(:,:)
       real, intent(out) :: dst(:,:)
     end subroutine inverse
  end interface

contains

  subroutine check(A)
    real, intent(inout) :: A(:,:)
    call inverse(A, A)
    ! CHECK-MESSAGES: :[[@LINE-1]]:5: warning: argument 'a' has mismatched intent
  end subroutine check
end module matrix
