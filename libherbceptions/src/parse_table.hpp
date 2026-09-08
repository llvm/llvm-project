// clang-format off
#define PARSE_ERRC_MAX_SIZE 14

	case 0:
		return __tsc(u8"Success");
	case 1:
		return __tsc(u8"End of file");
	case 2:
		return __tsc(u8"Partial parse");
	case 3:
		return __tsc(u8"Invalid format");
	case 4:
		return __tsc(u8"Overflow");
	default:
		return __tsc(u8"Unknown");
// clang-format on
