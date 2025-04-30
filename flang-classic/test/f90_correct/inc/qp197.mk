# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

build: $(SRC)/$(TEST).f08
	@echo ------------------------------------ building test $(TEST)
	-$(FC) -Mq,10,2 -c $(SRC)/$(TEST).f08 > $(TEST).rslt 2>&1

run:
	@echo ------------------------------------ nothing to run for test $(TEST)

verify: $(TEST).rslt
	@echo ------------------------------------ verifying test $(TEST)
	$(COMP_CHECK) $(SRC)/$(TEST).f08 $(TEST).rslt $(FC)
