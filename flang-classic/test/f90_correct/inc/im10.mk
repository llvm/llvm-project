#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test im10  ########


im10: run
	

build:  $(SRC)/im10.f
	-$(RM) im10.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/im10.f -o im10.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) im10.$(OBJX) check.$(OBJX) $(LIBS) -o im10.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test im10
	im10.$(EXESUFFIX)

verify: ;

im10.run: run

