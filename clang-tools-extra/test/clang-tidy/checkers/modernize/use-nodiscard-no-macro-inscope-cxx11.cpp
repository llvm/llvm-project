// RUN: %check_clang_tidy %s modernize-use-nodiscard %t -- \
// RUN:   -config="{CheckOptions: {modernize-use-nodiscard.ReplacementString: 'CUSTOM_NO_DISCARD'}}"

// As if the macro was not defined.
// #define CUSTOM_NO_DISCARD __attribute_((warn_unused_result))

class Foo
{
public:
    bool f1() const;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: function 'f1' should be marked CUSTOM_NO_DISCARD [modernize-use-nodiscard]
};

