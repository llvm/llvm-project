local:

.weak RegularWeak_with_RegularWeak
.size RegularWeak_with_RegularWeak, 32
RegularWeak_with_RegularWeak:

.global RegularWeak_with_RegularStrong
.size RegularWeak_with_RegularStrong, 33
RegularWeak_with_RegularStrong:

.weak RegularStrong_with_RegularWeak
.size RegularStrong_with_RegularWeak, 34
RegularStrong_with_RegularWeak:

.weak RegularWeak_with_UndefWeak
.size RegularWeak_with_UndefWeak, 35
.quad RegularWeak_with_UndefWeak

.size RegularWeak_with_UndefStrong, 36
.quad RegularWeak_with_UndefStrong

.weak RegularStrong_with_UndefWeak
.size RegularStrong_with_UndefWeak, 37
.quad RegularStrong_with_UndefWeak

.size RegularStrong_with_UndefStrong, 38
.quad RegularStrong_with_UndefStrong

.comm RegularWeak_with_CommonStrong,40,4

.comm RegularStrong_with_CommonStrong,42,4

.weak UndefWeak_with_RegularWeak
.size UndefWeak_with_RegularWeak, 43
UndefWeak_with_RegularWeak:

.global UndefWeak_with_RegularStrong
.size UndefWeak_with_RegularStrong, 44
UndefWeak_with_RegularStrong:

.weak UndefStrong_with_RegularWeak
.size UndefStrong_with_RegularWeak, 45
UndefStrong_with_RegularWeak:

.global UndefStrong_with_RegularStrong
.size UndefStrong_with_RegularStrong, 46
UndefStrong_with_RegularStrong:

.weak UndefWeak_with_UndefWeak
.size UndefWeak_with_UndefWeak, 47
.quad UndefWeak_with_UndefWeak

.comm UndefWeak_with_CommonStrong,49,4

.comm UndefStrong_with_CommonStrong,51,4

.weak CommonStrong_with_RegularWeak
.size CommonStrong_with_RegularWeak, 54
CommonStrong_with_RegularWeak:

.global CommonStrong_with_RegularStrong
.size CommonStrong_with_RegularStrong, 55
CommonStrong_with_RegularStrong:

.weak CommonStrong_with_UndefWeak
.size CommonStrong_with_UndefWeak, 58
.quad CommonStrong_with_UndefWeak

.size CommonStrong_with_UndefStrong, 59
.quad CommonStrong_with_UndefStrong

.comm CommonStrong_with_CommonStrong,63,4
