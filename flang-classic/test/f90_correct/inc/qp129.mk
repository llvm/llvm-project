#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test intrinsic  ########


qp129: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp129.f08 fcheck.$(OBJX)
	-$(RM) qp129.$(EXESUFFIX) qp129.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp129.f08 -o qp129.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp129.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp129.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp129
	qp129.$(EXESUFFIX)

verify: ;

qp129.run: run

