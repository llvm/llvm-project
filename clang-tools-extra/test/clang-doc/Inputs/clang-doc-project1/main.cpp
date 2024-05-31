#include <iostream>
#include "Calculator.h"
#include "Circle.h"
#include "Rectangle.h"

int main() {
    // Calculator
    Calculator calc;
    std::cout << "Add: " << calc.add(3, 4) << std::endl;
    std::cout << "Subtract: " << calc.subtract(10, 5) << std::endl;
    std::cout << "Multiply: " << calc.multiply(2, 3) << std::endl;
    std::cout << "Divide: " << calc.divide(10, 2) << std::endl;

    // Circle
    Circle circle(5.0);
    std::cout << "Circle Area: " << circle.area() << std::endl;
    std::cout << "Circle Perimeter: " << circle.perimeter() << std::endl;

    // Rectangle
    Rectangle rectangle(4.0, 6.0);
    std::cout << "Rectangle Area: " << rectangle.area() << std::endl;
    std::cout << "Rectangle Perimeter: " << rectangle.perimeter() << std::endl;

    return 0;
}