typedef enum __attribute__((enum_extensibility(closed))) ComparisonResult : long ComparisonResult; enum ComparisonResult : long {
    OrderedAscending = -1L,
    OrderedSame,
    OrderedDescending
};

ComparisonResult getReturn(long x) {
  switch (x) {
    case 0:
      return OrderedSame;
    case -1:
      return OrderedAscending;
    case 1:
    default:
      return OrderedDescending;
  }
}
