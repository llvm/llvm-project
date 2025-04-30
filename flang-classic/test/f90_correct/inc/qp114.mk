#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qtodb  ########


qp114: run


build:  $(SRC)/qp114.f08
	-$(RM) qp114.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o check_mod.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp114.f08 -o qp114.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp114.$(OBJX) check_mod.$(OBJX) $(LIBS) -o qp114.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp114
	qp114.$(EXESUFFIX)

verify: ;


