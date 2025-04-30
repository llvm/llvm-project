#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test il00  ########


il00: run
	

build:  $(SRC)/il00.f90
	-$(RM) il00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/il00.f90 -o il00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) il00.$(OBJX) check.$(OBJX) $(LIBS) -o il00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test il00
	il00.$(EXESUFFIX)

verify: ;

il00.run: run

