#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test il09  ########


il09: run
	

build:  $(SRC)/il09.f90
	-$(RM) il09.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/il09.f90 -o il09.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) il09.$(OBJX) check.$(OBJX) $(LIBS) -o il09.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test il09
	il09.$(EXESUFFIX)

verify: ;

il09.run: run

