template<typename T> struct DeductionGuide {};
DeductionGuide() -> DeductionGuide<int>;
