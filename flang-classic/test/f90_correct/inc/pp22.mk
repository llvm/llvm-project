#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pp22  ########


pp22: run
	

build:  $(SRC)/pp22.f90
	-$(RM) pp22.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pp22.f90 -o pp22.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pp22.$(OBJX) check.$(OBJX) $(LIBS) -o pp22.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pp22
	pp22.$(EXESUFFIX)

verify: ;

pp22.run: run

