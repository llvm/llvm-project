/* For some machines GNU MP needs to define an auxiliary function:

   udiv_qrnnd (quotient, remainder, high_numerator, low_numerator, denominator)

   Divides a two-word unsigned integer, composed by the integers
   HIGH_NUMERATOR and LOW_NUMERATOR, by DENOMINATOR and places the quotient
   in QUOTIENT and the remainder in REMAINDER.  HIGH_NUMERATOR must be less
   than DENOMINATOR for correct operation.  If, in addition, the most
   significant bit of DENOMINATOR must be 1, then the pre-processor symbol
   UDIV_NEEDS_NORMALIZATION is defined to 1.  */
