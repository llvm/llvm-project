#include "Rectangle.h"

Rectangle::Rectangle(double width, double height)
        : width_(width), height_(height) {}

double Rectangle::area() const {
    return width_ * height_;
}

double Rectangle::perimeter() const {
    return 2 * (width_ + height_);
}