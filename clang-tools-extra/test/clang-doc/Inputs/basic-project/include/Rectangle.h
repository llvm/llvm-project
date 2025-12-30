#pragma once

#include "Shape.h"

/**
 * @brief Rectangle class derived from Shape.
 *
 * Represents a rectangle with a given width and height.
 */
class Rectangle : public Shape {
public:
    /**
     * @brief Constructs a new Rectangle object.
     *
     * @param width Width of the rectangle.
     * @param height Height of the rectangle.
     */
    Rectangle(double width, double height);

    /**
     * @brief Calculates the area of the rectangle.
     *
     * @return double The area of the rectangle.
     */
    double area() const override;

    /**
     * @brief Calculates the perimeter of the rectangle.
     *
     * @return double The perimeter of the rectangle.
     */
    double perimeter() const override;

private:
    double width_; ///< Width of the rectangle.
    double height_; ///< Height of the rectangle.
};