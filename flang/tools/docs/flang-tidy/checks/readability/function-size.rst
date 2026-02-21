.. title:: flang-tidy - readability-function-size

readability-function-size
=========================

Enforces size constraints on functions and subroutines based on line count, nesting level, and number of dummy arguments. Default thresholds are 5 for nesting level and 5 for dummy arguments (line count threshold is configurable). This check helps maintain manageable and readable code by preventing procedures from growing too large or complex.

.. code-block:: fortran

    ! This will trigger warnings for excessive arguments and nesting
    subroutine process_data(data1, data2, data3, data4, data5, data6, options)
      implicit none
      real, intent(in) :: data1(:), data2(:), data3(:)
      real, intent(in) :: data4(:), data5(:), data6(:)
      integer, intent(in) :: options
      integer :: i, j, k, m, n

      do i = 1, size(data1)
        if (data1(i) > 0) then
          do j = 1, size(data2)
            if (data2(j) > data1(i)) then
              do k = 1, size(data3)
                if (data3(k) > data2(j)) then
                  do m = 1, size(data4)
                    if (data4(m) > data3(k)) then
                      do n = 1, size(data5)
                        if (data5(n) > data4(m)) then
                          ! This is at nesting level 10, which exceeds the threshold
                          print *, "Found match"
                        end if
                      end do
                    end if
                  end do
                end if
              end do
            end if
          end do
        end if
      end do
    end subroutine
