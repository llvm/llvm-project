 ! RUN: %check_flang_tidy %s bugprone-contiguous-array %t
module test
  interface
     subroutine contig(a)
       real, contiguous, intent(in) :: a(:)
     end subroutine contig

     ! CHECK-MESSAGES: :[[@LINE+2]]:28: warning: assumed-shape array 'a' should be contiguous
     subroutine possibly_noncontig(a)
       real, intent(in) :: a(:)
     end subroutine possibly_noncontig
  end interface
end module test
