// RUN: %check_clang_tidy %s modernize-use-nodiscard %t -- -- -std=c++17 

class Foo
{
public:
    bool f1() const;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: function 'f1' should be marked {{\[\[nodiscard\]\]}} [modernize-use-nodiscard]
    // CHECK-FIXES: {{\[\[nodiscard\]\]}} bool f1() const;

    bool f2(int) const;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: function 'f2' should be marked {{\[\[nodiscard\]\]}} [modernize-use-nodiscard]
    // CHECK-FIXES: {{\[\[nodiscard\]\]}} bool f2(int) const;

    bool f3(const int &) const;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: function 'f3' should be marked {{\[\[nodiscard\]\]}} [modernize-use-nodiscard]
    // CHECK-FIXES: {{\[\[nodiscard\]\]}} bool f3(const int &) const;

    bool f4(void) const;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: function 'f4' should be marked {{\[\[nodiscard\]\]}} [modernize-use-nodiscard]
    // CHECK-FIXES: {{\[\[nodiscard\]\]}} bool f4(void) const;

};
