! RUN: %check_flang_tidy %s readability-function-cognitive-complexity %t
MODULE function_size_test
  IMPLICIT NONE

CONTAINS
  SUBROUTINE simple_function(n, result)
    INTEGER, INTENT(IN) :: n
    INTEGER, INTENT(OUT) :: result
    INTEGER :: i, temp

    temp = 0
    DO i = 1, n
      IF (i > 5) THEN
        temp = temp + i
      END IF
    END DO

    result = temp
  END SUBROUTINE simple_function

  SUBROUTINE complex_function(arr, n, threshold, result)
    INTEGER, INTENT(IN) :: arr(:,:), n, threshold
    INTEGER, INTENT(OUT) :: result
    INTEGER :: i, j, temp, count

    temp = 0
    count = 0

    DO i = 1, n                                 ! Level 1
      DO j = 1, n                               ! Level 2
        IF (arr(i,j) > threshold) THEN          ! Level 3
          IF (arr(i,j) < threshold * 2) THEN    ! Level 4
            DO WHILE (temp < arr(i,j))          ! Level 5
              IF (temp > 0) THEN                ! Level 6
                temp = temp + 1
                count = count + 1
              END IF
            END DO
          END IF
        END IF
      END DO
    END DO

    result = count
  END SUBROUTINE complex_function

  ! CHECK-MESSAGES: :[[@LINE+1]]:3: warning: subroutine borderline_function(arr, n, result) has a cognitive complexity of 39, which exceeds the threshold of 25
  SUBROUTINE borderline_function(arr, n, result)
    INTEGER, INTENT(IN) :: arr(:), n
    INTEGER, INTENT(OUT) :: result
    INTEGER :: i, sum, count
    LOGICAL :: found

    sum = 0
    count = 0
    found = .FALSE.

    DO i = 1, n                          ! Level 1
      IF (arr(i) > 0) THEN               ! Level 2
        DO WHILE (.NOT. found)           ! Level 3
          IF (sum < 100) THEN            ! Level 4
            IF (arr(i) > 10) THEN        ! Level 5
              sum = sum + arr(i)
              count = count + 1
            END IF
          END IF

          found = (sum >= 100)
        END DO
      END IF
    END DO

    result = count
  END SUBROUTINE borderline_function

  ! CHECK-MESSAGES: :[[@LINE+1]]:3: warning: subroutine extremely_nested_function(x, result) has a cognitive complexity of 61, which exceeds the threshold of 25
  SUBROUTINE extremely_nested_function(x, result)
    INTEGER, INTENT(IN) :: x
    INTEGER, INTENT(OUT) :: result
    INTEGER :: i, j, k

    result = 0

    outer: DO i = 1, 10                        ! Level 1
      middle: DO j = 1, 10                     ! Level 2
        inner: DO k = 1, 10                    ! Level 3
          IF (i*j*k > x) THEN                  ! Level 4
            IF (i*j*k < x*2) THEN              ! Level 5
              BLOCK                            ! Level 6
                IF (MOD(i*j*k, 2) == 0) THEN   ! Level 7
                  result = result + 1
                END IF
              END BLOCK
            END IF
          END IF
        END DO inner
      END DO middle
    END DO outer

  END SUBROUTINE extremely_nested_function

  ! CHECK-MESSAGES: :[[@LINE+1]]:3: warning: subroutine case_function(x, y, result) has a cognitive complexity of 71, which exceeds the threshold of 25
  SUBROUTINE case_function(x, y, result)
    INTEGER, INTENT(IN) :: x, y
    INTEGER, INTENT(OUT) :: result
    INTEGER :: i

    result = 0

    DO i = 1, 10                           ! Level 1
      SELECT CASE (i)                      ! Level 2
        CASE (1:3)
          IF (x > 0) THEN                  ! Level 3
            SELECT CASE (y)                ! Level 4
              CASE (1:5)
                result = result + 1
              CASE DEFAULT
                result = result + 2
            END SELECT
          END IF
        CASE DEFAULT
          result = result + 3
      END SELECT
    END DO

  END SUBROUTINE case_function

END MODULE function_size_test
