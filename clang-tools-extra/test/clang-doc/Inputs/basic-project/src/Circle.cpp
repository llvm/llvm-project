#include "Circle.h"

Circle::Circle(double radius) : radius_(radius) {}

double Circle::area() const {
    return 3.141 * radius_ * radius_;
}

double Circle::perimeter() const {
    return 3.141 * radius_;
}