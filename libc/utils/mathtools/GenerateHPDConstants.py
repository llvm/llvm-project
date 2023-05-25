from math import *

"""
This script is used to generate a table used by 
libc/src/__support/high_precision_decimal.h.

For the ith entry in the table there are two values (indexed starting at 0).
The first value is the number of digits longer the second value would be if
multiplied by 2^i.
The second value is the smallest number that would create that number of 
additional digits (which in base ten is always 5^i). Anything less creates one 
fewer digit.

As an example, the 3rd entry in the table is {1, "125"}. This means that if 
125 is multiplied by 2^3 = 8, it will have exactly one more digit.
Multiplying it out we get 125 * 8 = 1000. 125 is the smallest number that gives
that extra digit, for example 124 * 8 = 992, and all larger 3 digit numbers
also give only one extra digit when multiplied by 8, for example 8 * 999 = 7992.
This makes sense because 5^3 * 2^3 = 10^3, the smallest 4 digit number.

For numbers with more digits we can ignore the digits past what's in the second
value, since the most significant digits determine how many extra digits there 
will be. Looking at the previous example, if we have 1000, and we look at just 
the first 3 digits (since 125 has 3 digits), we see that 100 < 125, so we get 
one fewer than 1 extra digits, which is 0. 
Multiplying it out we get 1000 * 8 = 8000, which fits the expectation. 
Another few quick examples: 
For 1255, 125 !< 125, so 1 digit more: 1255 * 8 = 10040
For 9999, 999 !< 125, so 1 digit more: 9999 * 8 = 79992

Now let's try an example with the 10th entry: {4, "9765625"}. This one means 
that 9765625 * 2^10 will have 4 extra digits. 
Let's skip straight to the examples:
For 1, 1 < 9765625, so 4-1=3 extra digits: 1 * 2^10 = 1024, 1 digit to 4 digits is a difference of 3.
For 9765624, 9765624 < 9765625 so 3 extra digits: 9765624 * 1024 = 9999998976, 7 digits to 10 digits is a difference of 3.
For 9765625, 9765625 !< 9765625 so 4 extra digits: 9765625 * 1024 = 10000000000, 7 digits to 11 digits is a difference of 4.
For 9999999, 9999999 !< 9765625 so 4 extra digits: 9999999 * 1024 = 10239998976, 7 digits to 11 digits is a difference of 4.
For 12345678, 1234567 < 9765625 so 3 extra digits: 12345678 * 1024 = 12641974272, 8 digits to 11 digits is a difference of 3.


This is important when left shifting in the HPD class because it reads and
writes the number backwards, and to shift in place it needs to know where the
last digit should go. Since a binary left shift by i bits is the same as
multiplying by 2^i we know that looking up the ith element in the table will
tell us the number of additional digits. If the first digits of the number
being shifted are greater than or equal to the digits of 5^i (the second value
of each entry) then it is just the first value in the entry, else it is one
fewer.
"""


# Generate Left Shift Table
outStr = ""
for i in range(61):
    tenToTheI = 10**i
    fiveToTheI = 5**i
    outStr += "{"
    # The number of new digits that would be created by multiplying 5**i by 2**i
    outStr += str(ceil(log10(tenToTheI) - log10(fiveToTheI)))
    outStr += ', "'
    if not i == 0:
        outStr += str(fiveToTheI)
    outStr += '"},\n'

print(outStr)
