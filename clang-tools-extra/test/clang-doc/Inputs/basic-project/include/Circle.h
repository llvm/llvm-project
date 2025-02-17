#pragma once

#include "Shape.h"

/**
 * @brief Circle class derived from Shape.
 *
 * Represents a circle with a given radius.
 */
class Circle : public Shape {
public:
    /**
     * @brief Constructs a new Circle object.
     *
     * @param radius Radius of the circle.
     */
    Circle(double radius);

    /**
     * @brief Calculates the area of the circle.
     *
     * @return double The area of the circle.
     */
    double area() const override;

    /**
     * @brief Calculates the perimeter of the circle.
     *
     * @return double The perimeter of the circle.
     */
    double perimeter() const override;

private:
    double radius_; ///< Radius of the circle.
};
