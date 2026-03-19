! RUN: %python %S/test_folding.py %s %flang_fc1
character(10), parameter :: a = '0123456789'
character(3), parameter :: arr(3) = [(a(1:i), i=1,3)]
logical, parameter :: test1 = all(arr == ["0", "01", "012"])
end
