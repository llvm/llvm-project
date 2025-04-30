#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pp02  ########


pp02: run
	

build:  $(SRC)/pp02.f90
	-$(RM) pp02.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pp02.f90 -o pp02.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pp02.$(OBJX) check.$(OBJX) $(LIBS) -o pp02.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pp02
	pp02.$(EXESUFFIX)

verify: ;

pp02.run: run

