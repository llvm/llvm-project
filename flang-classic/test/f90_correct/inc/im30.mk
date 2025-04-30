#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test im30  ########


im30: run
	

build:  $(SRC)/im30.f
	-$(RM) im30.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/im30.f -o im30.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) im30.$(OBJX) check.$(OBJX) $(LIBS) -o im30.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test im30
	im30.$(EXESUFFIX)

verify: ;

im30.run: run

