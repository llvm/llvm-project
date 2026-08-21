class Virtual {};
class Foo : virtual Virtual {};
class Bar : Foo {};
class Fizz : virtual Virtual {};
class Buzz : Fizz {};

class MyClass : Bar, Buzz {};
