#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

OBJ=external_pointer_common.$(OBJX)

build:  $(SRC)/external_pointer_common.f90
	-$(RM) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	-$(RM) $(OBJ)
	@echo ------------------------------------ building test $@
	$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/external_pointer_common.f90 -o $(OBJ)

run:
	@echo nothing to run

verify: $(OBJ)
	@echo PASS

