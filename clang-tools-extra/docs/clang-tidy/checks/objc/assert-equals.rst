.. title:: clang-tidy - objc-assert-equals

objc-assert-equals
==================

Finds improper usages of `XCTAssertEqual` and `XCTAssertNotEqual` and replaces
them with `XCTAssertEqualObjects` or `XCTAssertNotEqualObjects`.

This makes tests less fragile, as many improperly rely on pointer equality for
strings that have equal values.  This assumption is not guaranteed by the
language.
