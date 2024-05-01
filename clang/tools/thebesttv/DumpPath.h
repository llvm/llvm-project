#pragma once

#include "utils.h"

bool saveLocationInfo(ASTContext &Context, const SourceRange &range,
                      ordered_json &j);
