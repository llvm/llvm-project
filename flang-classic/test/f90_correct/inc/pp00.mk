#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pp00  ########


pp00: run
	

build:  $(SRC)/pp00.f90
	-$(RM) pp00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pp00.f90 -o pp00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pp00.$(OBJX) check.$(OBJX) $(LIBS) -o pp00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pp00
	pp00.$(EXESUFFIX)

verify: ;

pp00.run: run

