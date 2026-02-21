.. title:: flang-tidy - readability-function-cognitive-complexity

readability-function-cognitive-complexity
=========================================

Verifies that the cognitive complexity of a function or subroutine does not exceed a threshold (default: 25). Cognitive complexity is a measure of how difficult code is to understand based on nesting, control flow, and logical operations. This check helps maintain readable and maintainable code by identifying overly complex procedures.

.. code-block:: fortran

    ! This will trigger a warning due to high cognitive complexity
    subroutine complex_procedure(arr, n, threshold)
      implicit none
      integer, intent(in) :: n, threshold
      real, intent(inout) :: arr(n)
      integer :: i, j, count

      count = 0
      do i = 1, n
        if (arr(i) > threshold) then
          do j = i, n
            if (arr(j) > 0) then
              if (arr(j) > arr(i)) then
                arr(j) = arr(j) - arr(i)
                if (arr(j) < threshold .and. arr(j) > 0) then
                  count = count + 1
                  if (count > n/2) then
                    do while (arr(j) > 0.1)
                      arr(j) = arr(j) * 0.9
                    end do
                  else if (count > n/3) then
                    arr(j) = arr(j) * 0.8
                  end if
                end if
              end if
            end if
          end do
        else if (arr(i) < -threshold) then
          do j = 1, i
            if (arr(j) < 0 .and. arr(j) < arr(i)) then
              arr(j) = arr(j) - arr(i)
            end if
          end do
        end if
      end do
    end subroutine
