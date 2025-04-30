#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ka38  ########


ka38: run
	

build:  $(SRC)/ka38.f
	-$(RM) ka38.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ka38.f -o ka38.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ka38.$(OBJX) check.$(OBJX) $(LIBS) -o ka38.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ka38
	ka38.$(EXESUFFIX)

verify: ;

ka38.run: run

