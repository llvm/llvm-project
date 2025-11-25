//! Google Search grounding support

pub struct Grounding {
    pub enabled: bool,
}

impl Grounding {
    pub fn new() -> Self {
        Self { enabled: false }
    }

    pub fn enable(&mut self) {
        self.enabled = true;
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }

    pub fn toggle(&mut self) {
        self.enabled = !self.enabled;
    }
}

impl Default for Grounding {
    fn default() -> Self {
        Self::new()
    }
}
