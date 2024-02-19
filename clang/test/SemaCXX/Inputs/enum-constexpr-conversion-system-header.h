// System header for testing that -Wenum-constexpr-conversion leads to an error
// when included in user code, or when the system macro is used.

enum SystemEnum
{
    a = 0,
    b = 1,
};

void testValueInRangeOfEnumerationValuesInSystemHeader()
{
    constexpr SystemEnum x1 = static_cast<SystemEnum>(123);
    // expected-error@-1 {{integer value 123 is outside the valid range of values [0, 1] for the enumeration type 'SystemEnum'}}

    const SystemEnum x2 = static_cast<SystemEnum>(123);  // ok, not a constant expression context
}

#define CONSTEXPR_CAST_TO_SYSTEM_ENUM_OUTSIDE_OF_RANGE \
    constexpr SystemEnum system_enum = static_cast<SystemEnum>(123)
