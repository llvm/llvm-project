// clang-format off
#define WINE_ERRC_MAX_SIZE 47

	case 0:
		return __tsc(u8"No error");
	case 1:
		return __tsc(u8"Operation not permitted");
	case 2:
		return __tsc(u8"No such file or directory");
	case 3:
		return __tsc(u8"No such process");
	case 4:
		return __tsc(u8"Interrupted system call");
	case 5:
		return __tsc(u8"I/O error");
	case 6:
		return __tsc(u8"No such device or address");
	case 7:
		return __tsc(u8"Argument list too long");
	case 8:
		return __tsc(u8"Exec format error");
	case 9:
		return __tsc(u8"Bad file number");
	case 10:
		return __tsc(u8"No child processes");
	case 11:
		return __tsc(u8"Try again");
	case 12:
		return __tsc(u8"Out of memory");
	case 13:
		return __tsc(u8"Permission denied");
	case 14:
		return __tsc(u8"Bad address");
	case 15:
		return __tsc(u8"Block device required");
	case 16:
		return __tsc(u8"Device or resource busy");
	case 17:
		return __tsc(u8"File exists");
	case 18:
		return __tsc(u8"Cross-device link");
	case 19:
		return __tsc(u8"No such device");
	case 20:
		return __tsc(u8"Not a directory");
	case 21:
		return __tsc(u8"Is a directory");
	case 22:
		return __tsc(u8"Invalid argument");
	case 23:
		return __tsc(u8"File table overflow");
	case 24:
		return __tsc(u8"Too many open files");
	case 25:
		return __tsc(u8"Not a typewriter");
	case 26:
		return __tsc(u8"Text file busy");
	case 27:
		return __tsc(u8"File too large");
	case 28:
		return __tsc(u8"No space left on device");
	case 29:
		return __tsc(u8"Illegal seek");
	case 30:
		return __tsc(u8"Read-only file system");
	case 31:
		return __tsc(u8"Too many links");
	case 32:
		return __tsc(u8"Broken pipe");
	case 33:
		return __tsc(u8"Math argument out of domain of func");
	case 34:
		return __tsc(u8"Math result not representable");
	case 35:
		return __tsc(u8"Resource deadlock would occur");
	case 36:
		return __tsc(u8"File name too long");
	case 37:
		return __tsc(u8"No record locks available");
	case 38:
		return __tsc(u8"Invalid system call number");
	case 39:
		return __tsc(u8"Directory not empty");
	case 40:
		return __tsc(u8"Too many symbolic links encountered");
	case 41:
		return __tsc(u8"Operation would block");
	case 42:
		return __tsc(u8"No message of desired type");
	case 43:
		return __tsc(u8"Identifier removed");
	case 44:
		return __tsc(u8"Channel number out of range");
	case 45:
		return __tsc(u8"Level 2 not synchronized");
	case 46:
		return __tsc(u8"Level 3 halted");
	case 47:
		return __tsc(u8"Level 3 reset");
	case 48:
		return __tsc(u8"Link number out of range");
	case 49:
		return __tsc(u8"Protocol driver not attached");
	case 50:
		return __tsc(u8"No CSI structure available");
	case 51:
		return __tsc(u8"Level 2 halted");
	case 52:
		return __tsc(u8"Invalid exchange");
	case 53:
		return __tsc(u8"Invalid request descriptor");
	case 54:
		return __tsc(u8"Exchange full");
	case 55:
		return __tsc(u8"No anode");
	case 56:
		return __tsc(u8"Invalid request code");
	case 57:
		return __tsc(u8"Invalid slot");
	case 58:
		return __tsc(u8"Edeadlock");
	case 59:
		return __tsc(u8"Bad font file format");
	case 60:
		return __tsc(u8"Device not a stream");
	case 61:
		return __tsc(u8"No data available");
	case 62:
		return __tsc(u8"Timer expired");
	case 63:
		return __tsc(u8"Out of streams resources");
	case 64:
		return __tsc(u8"Machine is not on the network");
	case 65:
		return __tsc(u8"Package not installed");
	case 66:
		return __tsc(u8"Object is remote");
	case 67:
		return __tsc(u8"Link has been severed");
	case 68:
		return __tsc(u8"Advertise error");
	case 69:
		return __tsc(u8"Srmount error");
	case 70:
		return __tsc(u8"Communication error on send");
	case 71:
		return __tsc(u8"Protocol error");
	case 72:
		return __tsc(u8"Multihop attempted");
	case 73:
		return __tsc(u8"RFS specific error");
	case 74:
		return __tsc(u8"Not a data message");
	case 75:
		return __tsc(u8"Value too large for defined data type");
	case 76:
		return __tsc(u8"Name not unique on network");
	case 77:
		return __tsc(u8"File descriptor in bad state");
	case 78:
		return __tsc(u8"Remote address changed");
	case 79:
		return __tsc(u8"Can not access a needed shared library");
	case 80:
		return __tsc(u8"Accessing a corrupted shared library");
	case 81:
		return __tsc(u8".lib section in a.out corrupted");
	case 82:
		return __tsc(u8"Attempting to link in too many shared libraries");
	case 83:
		return __tsc(u8"Cannot exec a shared library directly");
	case 84:
		return __tsc(u8"Illegal byte sequence");
	case 85:
		return __tsc(u8"Interrupted system call should be restarted");
	case 86:
		return __tsc(u8"Streams pipe error");
	case 87:
		return __tsc(u8"Too many users");
	case 88:
		return __tsc(u8"Socket operation on non-socket");
	case 89:
		return __tsc(u8"Destination address required");
	case 90:
		return __tsc(u8"Message too long");
	case 91:
		return __tsc(u8"Protocol wrong type for socket");
	case 92:
		return __tsc(u8"Protocol not available");
	case 93:
		return __tsc(u8"Protocol not supported");
	case 94:
		return __tsc(u8"Socket type not supported");
	case 95:
		return __tsc(u8"Operation not supported on transport endpoint");
	case 96:
		return __tsc(u8"Protocol family not supported");
	case 97:
		return __tsc(u8"Address family not supported by protocol");
	case 98:
		return __tsc(u8"Address already in use");
	case 99:
		return __tsc(u8"Cannot assign requested address");
	case 100:
		return __tsc(u8"Network is down");
	case 101:
		return __tsc(u8"Network is unreachable");
	case 102:
		return __tsc(u8"Network dropped connection because of reset");
	case 103:
		return __tsc(u8"Software caused connection abort");
	case 104:
		return __tsc(u8"Connection reset by peer");
	case 105:
		return __tsc(u8"No buffer space available");
	case 106:
		return __tsc(u8"Transport endpoint is already connected");
	case 107:
		return __tsc(u8"Transport endpoint is not connected");
	case 108:
		return __tsc(u8"Cannot send after transport endpoint shutdown");
	case 109:
		return __tsc(u8"Too many references: cannot splice");
	case 110:
		return __tsc(u8"Connection timed out");
	case 111:
		return __tsc(u8"Connection refused");
	case 112:
		return __tsc(u8"Host is down");
	case 113:
		return __tsc(u8"No route to host");
	case 114:
		return __tsc(u8"Operation already in progress");
	case 115:
		return __tsc(u8"Operation now in progress");
	case 116:
		return __tsc(u8"Stale file handle");
	case 117:
		return __tsc(u8"Structure needs cleaning");
	case 118:
		return __tsc(u8"Not a XENIX named type file");
	case 119:
		return __tsc(u8"No XENIX semaphores available");
	case 120:
		return __tsc(u8"Is a named type file");
	case 121:
		return __tsc(u8"Remote I/O error");
	case 122:
		return __tsc(u8"Quota exceeded");
	case 123:
		return __tsc(u8"No medium found");
	case 124:
		return __tsc(u8"Wrong medium type");
	case 125:
		return __tsc(u8"Operation Canceled");
	case 126:
		return __tsc(u8"Required key not available");
	case 127:
		return __tsc(u8"Key has expired");
	case 128:
		return __tsc(u8"Key has been revoked");
	case 129:
		return __tsc(u8"Key was rejected by service");
	case 130:
		return __tsc(u8"Owner died");
	case 131:
		return __tsc(u8"State not recoverable");
	case 132:
		return __tsc(u8"Operation not possible due to RF-kill");
	case 133:
		return __tsc(u8"Memory page has hardware error");
	default:
		return __tsc(u8"Unknown");
// clang-format on
