@protocol NSObject
- (oneway void)release;
@end

#ifdef PART1
static inline void part1(id p) {
  [p release];
}
#endif

#ifdef PART2
static inline void part2(id p) {
  [p release];
}
#endif
