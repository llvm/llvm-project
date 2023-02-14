! RUN: %python %S/test_errors.py %s %flang_fc1
character(4) a, b, c, d, e
!WARNING: DATA statement value '"abcde"' for 'a' has the wrong length
data a(1:4)/'abcde'/
!WARNING: DATA statement value '"abc"' for 'b' has the wrong length
data b(1:4)/'abc'/
data c/'abcde'/ ! not a substring, conforms
data d/'abc'/ ! not a substring, conforms
!ERROR: DATA statement designator 'e(1_8:5_8)' is out of range
data e(1:5)/'xyz'/
end
