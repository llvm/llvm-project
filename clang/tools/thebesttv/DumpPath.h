#pragma once

#include "utils.h"

void saveLocationInfo(ASTContext &Context, const SourceRange &range,
                      ordered_json &j);
